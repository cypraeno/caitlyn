#include "render.h"

void setRenderData(RenderData& render_data, const float aspect_ratio, const int image_width, const int samples_per_pixel, const int max_depth) {
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    render_data.image_width = image_width;
    render_data.image_height = image_height;
    render_data.samples_per_pixel = samples_per_pixel;
    render_data.max_depth = max_depth;
    render_data.buffer = std::vector<color>(image_width * image_height);
}

color colorize_ray(const ray& r, std::shared_ptr<Scene> scene, int depth) {
    HitInfo record;

    // end of recursion
    if (depth <= 0) {
        return color(0,0,0);
    }

    // fire ray into scene and get ID.
    struct RTCRayHit rayhit;
    setupRayHit1(rayhit, r);

    rtcIntersect1(scene->rtc_scene, &rayhit);

    int targetID;
    if (rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID) { // hit an instance
        targetID = rayhit.hit.instID[0];
    } else if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        targetID = rayhit.hit.geomID;
    } else {
        // Sky background (gradient blue-white)
        vec3 unit_direction = r.direction().unit_vector();
        auto t = 0.5*(unit_direction.y() + 1.0);

        return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
    }

    // Hit is found
    ray scattered;
    color attenuation;

    // get the material of the thing we just hit
    std::shared_ptr<Geometry> geomhit = scene->geom_map[targetID];
    std::shared_ptr<material> mat_ptr = geomhit->materialById(targetID);
    record = geomhit->getHitInfo(r, r.at(rayhit.ray.tfar), rayhit.ray.tfar, targetID);

    color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);
    if (!mat_ptr->scatter(r, record, attenuation, scattered)) {
        return color_from_emission;
    } 

    color color_from_scatter = attenuation * colorize_ray(scattered, scene, depth-1);

    return color_from_emission + color_from_scatter;
}

void render_scanlines(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam) {

    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;

    for (int j=start_line; j>=start_line - (lines - 1); --j) {

        for (int i=0; i<image_width; ++i) {

            color pixel_color(0, 0, 0);

            for (int s=0; s < samples_per_pixel; s++) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += colorize_ray(r, scene_ptr, max_depth);
            }

            int buffer_index = j * image_width + i;
            color buffer_pixel(pixel_color.x(), pixel_color.y(), pixel_color.z());
            data.buffer[buffer_index] = buffer_pixel;
        }
        data.completed_lines += 1;

        float percentage_completed = ((float)data.completed_lines / (float)data.image_height)*100.00;
        std::cerr << "[" <<int(percentage_completed) << "%] completed" << std::endl;
    }
}

void completeRayQueueTask(std::vector<RayQueue>& current, std::vector<color>& temp_buffer,
                            std::vector<color>& full_buffer, std::vector<RayQueue>& queue,
                            int mask[], int i, int current_index) {
    // check if theres even any more to do, if not then break out.
    // this pixel is done so we can update the full buffer.
    full_buffer[current_index] += temp_buffer[current_index];
    if (queue.empty()) {
        mask[i] = 0; // disable this part of the packet from running
    } else {
        // replace finished RayQueue with next
        RayQueue back = queue.back();
        queue.pop_back();
        current[i] = back;
    }
}

void render_scanlines_sse(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam) {
    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;

    std::vector<color> full_buffer(image_width);

    std::vector<RayQueue> queue;
    queue.reserve(image_width);

    std::vector<color> temp_buffer(image_width);
    std::vector<color> attenuation_buffer(image_width);
    std::vector<RayQueue> current(4); // size = 4 only

    int mask[4] = {-1, -1, -1, -1};
    
    for (int j=start_line; j>=start_line - (lines - 1); --j) {
        std::fill(full_buffer.begin(), full_buffer.end(), color(0, 0, 0));
        for (int s=0; s < samples_per_pixel; s++) {
            std::fill(temp_buffer.begin(), temp_buffer.end(), color(0, 0, 0));
            std::fill(attenuation_buffer.begin(), attenuation_buffer.end(), color(0, 0, 0));
            queue.clear();
            for (int i=image_width-1; i>=0; --i) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                RayQueue q = { i, 0, r };
                queue.push_back(q);
            }

            RTCRayHit4 rayhit;

            for (int i=0; i<4; i++) {
                RayQueue back = queue.back();
                queue.pop_back();
                current[i] = back;
            }

            std::fill(std::begin(mask), std::end(mask), -1);
            while (mask[0] != 0 or mask[1] != 0 or mask[2] != 0 or mask[3] != 0) {
                std::vector<ray> rays;
                for (int i=0; i<(int)current.size(); i++) {
                    rays.push_back(current[i].r);
                }
                setupRayHit4(rayhit, rays);
                rtcIntersect4(mask, scene_ptr->rtc_scene, &rayhit);

                HitInfo record;

                for (int i=0; i<4; i++) {
                    if (mask[i] == 0) { continue; }
                    ray current_ray = current[i].r;
                    int current_index = current[i].index;

                    // process each ray by editing the temp_buffer and updating current queue
                    int targetID = -1;
                    if (rayhit.hit.instID[0][i] != RTC_INVALID_GEOMETRY_ID) { 
                        targetID = rayhit.hit.instID[0][i]; }
                    else if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) {
                        targetID = rayhit.hit.geomID[i]; }
                    else { // no hit
                        // Sky background (gradient blue-white)
                        vec3 unit_direction = current_ray.direction().unit_vector();
                        auto t = 0.5*(unit_direction.y() + 1.0);

                        color multiplier = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
                        if (current[i].depth == 0) { temp_buffer[current_index] = multiplier; }
                        else { temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * multiplier); }
                        completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                    }
                    if (targetID != -1) {
                        ray scattered;
                        color attenuation;
                        std::shared_ptr<Geometry> geomhit = scene_ptr->geom_map[targetID];
                        std::shared_ptr<material> mat_ptr = geomhit->materialById(targetID);
                        record = geomhit->getHitInfo(current_ray, current_ray.at(rayhit.ray.tfar[i]), rayhit.ray.tfar[i], targetID);
                        
                        color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);
                        if (!mat_ptr->scatter(current_ray, record, attenuation, scattered)) {
                            if (current[i].depth == 0) { temp_buffer[current_index] = color_from_emission; }
                            else { temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * color_from_emission); }
                            completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                        } else {
                            if (current[i].depth == 0) {
                                temp_buffer[current_index] = color_from_emission;
                                attenuation_buffer[current_index] = attenuation;
                            }
                            else {
                                temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * color_from_emission);
                                attenuation_buffer[current_index] = attenuation_buffer[current_index] * attenuation;
                            }
                            if (current[i].depth + 1 == max_depth) { // reached max depth, replace with next in queue
                                completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                            } else { // not finished depth wise
                                current[i].depth += 1;
                                current[i].r = scattered;
                            }
                        }
                    }
                }
            }
        }
        for (int i=0; i<image_width; ++i) {
            int buffer_index = j * image_width + i;
            data.buffer[buffer_index] = color(full_buffer[i].x(), full_buffer[i].y(), full_buffer[i].z());
        }
        data.completed_lines += 1;
        float percentage_completed = ((float)data.completed_lines / (float)data.image_height)*100.00;
        std::cerr << "[" <<int(percentage_completed) << "%] completed" << std::endl;
    }
}

void render_scanlines_avx(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam) {
    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;

    std::vector<color> full_buffer(image_width);

    std::vector<RayQueue> queue;
    queue.reserve(image_width);

    std::vector<color> temp_buffer(image_width);
    std::vector<color> attenuation_buffer(image_width);
    std::vector<RayQueue> current(8); // size = 8 only

    int mask[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    
    for (int j=start_line; j>=start_line - (lines - 1); --j) {
        std::fill(full_buffer.begin(), full_buffer.end(), color(0, 0, 0));
        for (int s=0; s < samples_per_pixel; s++) {
            std::fill(temp_buffer.begin(), temp_buffer.end(), color(0, 0, 0));
            std::fill(attenuation_buffer.begin(), attenuation_buffer.end(), color(0, 0, 0));
            queue.clear();
            for (int i=image_width-1; i>=0; --i) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                RayQueue q = { i, 0, r };
                queue.push_back(q);
            }

            RTCRayHit8 rayhit;

            for (int i=0; i<8; i++) {
                RayQueue back = queue.back();
                queue.pop_back();
                current[i] = back;
            }

            std::fill(std::begin(mask), std::end(mask), -1);
            while (mask[0] != 0 or mask[1] != 0 or mask[2] != 0 or mask[3] != 0
                    or mask[4] != 0 or mask[5] != 0 or mask[6] != 0 or mask[7] != 0) {
                std::vector<ray> rays;
                for (int i=0; i<(int)current.size(); i++) {
                    rays.push_back(current[i].r);
                }
                setupRayHit8(rayhit, rays);
                rtcIntersect8(mask, scene_ptr->rtc_scene, &rayhit);

                HitInfo record;

                for (int i=0; i<8; i++) {
                    if (mask[i] == 0) { continue; }
                    ray current_ray = current[i].r;
                    int current_index = current[i].index;

                    // process each ray by editing the temp_buffer and updating current queue
                    int targetID = -1;
                    if (rayhit.hit.instID[0][i] != RTC_INVALID_GEOMETRY_ID) { 
                        targetID = rayhit.hit.instID[0][i]; }
                    else if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) {
                        targetID = rayhit.hit.geomID[i]; }
                    else { // no hit
                        // Sky background (gradient blue-white)
                        vec3 unit_direction = current_ray.direction().unit_vector();
                        auto t = 0.5*(unit_direction.y() + 1.0);

                        color multiplier = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
                        if (current[i].depth == 0) { temp_buffer[current_index] = multiplier; }
                        else { temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * multiplier); }
                        completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                    }

                    if (targetID != -1) {
                        ray scattered;
                        color attenuation;
                        std::shared_ptr<Geometry> geomhit = scene_ptr->geom_map[rayhit.hit.geomID[i]];
                        std::shared_ptr<material> mat_ptr = geomhit->materialById(rayhit.hit.geomID[i]);
                        record = geomhit->getHitInfo(current_ray, current_ray.at(rayhit.ray.tfar[i]), rayhit.ray.tfar[i], rayhit.hit.geomID[i]);
                        
                        color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);
                        if (!mat_ptr->scatter(current_ray, record, attenuation, scattered)) {
                            if (current[i].depth == 0) { temp_buffer[current_index] = color_from_emission; }
                            else { temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * color_from_emission); }
                            completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                        } else {
                            if (current[i].depth == 0) {
                                temp_buffer[current_index] = color_from_emission;
                                attenuation_buffer[current_index] = attenuation;
                            }
                            else {
                                temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * color_from_emission);
                                attenuation_buffer[current_index] = attenuation_buffer[current_index] * attenuation;
                            }
                            if (current[i].depth + 1 == max_depth) { // reached max depth, replace with next in queue
                                completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                            } else { // not finished depth wise
                                current[i].depth += 1;
                                current[i].r = scattered;
                            }
                        }
                    }
                }
            }
        }
        for (int i=0; i<image_width; ++i) {
            int buffer_index = j * image_width + i;
            data.buffer[buffer_index] = color(full_buffer[i].x(), full_buffer[i].y(), full_buffer[i].z());
        }
        data.completed_lines += 1;
        float percentage_completed = ((float)data.completed_lines / (float)data.image_height)*100.00;
        std::cerr << "[" <<int(percentage_completed) << "%] completed" << std::endl;
    }
}
