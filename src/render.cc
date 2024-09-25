#include "render.h"

color trace_ray(const ray& r, std::shared_ptr<Scene> scene, int depth) {
    HitInfo record;

    color weight = color(1.0, 1.0, 1.0);
    color accumulated_color = color(0,0,0);

    ray r_in = r;
    BSDF_TYPE incoming_type = BSDF_TYPE::DIFFUSE;

    MediumRecord med_rec(r.origin());
    // Trace rays in all volume scenes to check which ones we reside in
    struct RTCRayHit vol_rayhit;
    HitInfo vol_record;
    for (const auto& ptr : scene->volumes) {
        setupRayHit1(vol_rayhit, r_in);
        rtcIntersect1(ptr->volume_scene, &vol_rayhit);
        int targetID;
        if (vol_rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID) {
            targetID = vol_rayhit.hit.instID[0];
        } else if (vol_rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            targetID = vol_rayhit.hit.geomID;
        } else {
            continue;
        }
        vol_record = ptr->getHitInfo(r_in, r_in.at(vol_rayhit.ray.tfar), vol_rayhit.ray.tfar, targetID);
        if (!vol_record.front_face) { // inside the volume!
            record.medium = false;
            record = vol_record;
            if (record.medium) {
                med_rec.hitVolume(ptr->medium, r_in.origin());
                incoming_type = BSDF_TYPE::SPECULAR;
            }
        }
    }

    for (int i=0; i<depth; i++) {
        // Enable of disable direct light sampling (debug only, should always be enabled)
        bool direct = true;
        bool raymarched = false; // set to true if we are colliding with a medium particle and not a surface
        std::shared_ptr<material> mat_ptr = nullptr;
        ray scattered;
        color attenuation;
        struct RTCRayHit rayhit;
        setupRayHit1(rayhit, r_in);

        rtcIntersect1(scene->rtc_scene, &rayhit);

        // Check for volume intersections
        float particleDist;
        shared_ptr<Medium> m_ptr = med_rec.particleDistance(particleDist);
        if (m_ptr) { // we are in a medium
            float istDist = rayhit.ray.tfar * r_in.direction().length();
            if (particleDist < istDist) { // volume intersection
                // Update record
                record.pos = r_in.at(particleDist / r_in.direction().length());
                raymarched = true;
            }
        }

        if (!raymarched) { // process information of next geometry hit
            int targetID;
            if (rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID) {
                targetID = rayhit.hit.instID[0];
            } else if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                targetID = rayhit.hit.geomID;
            } else {
                // Sky background (gradient blue-white)
                vec3 unit_direction = r_in.direction().unit_vector();
                auto t = 0.5*(unit_direction.y() + 1.0);

                // color sky = color(0,0,0);
                color sky = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
                accumulated_color += weight * sky;
                return accumulated_color;
            }

            std::shared_ptr<Geometry> geomhit = scene->geom_map[targetID];
            mat_ptr = geomhit->materialById(targetID);
            record.medium = false;
            record = geomhit->getHitInfo(r_in, r_in.at(rayhit.ray.tfar), rayhit.ray.tfar, targetID);
            if (record.medium) {
                std::shared_ptr<Volume> volhit = std::dynamic_pointer_cast<Volume>(geomhit);
                if (volhit) {
                    med_rec.hitVolume(volhit->medium, r_in.at(rayhit.ray.tfar));
                    r_in = ray(r_in.at(rayhit.ray.tfar), r_in.direction(), 0.0);
                    incoming_type = BSDF_TYPE::SPECULAR;
                    continue; // ignore edges of volumes? move to next bounce
                }
            }
        } else {
            mat_ptr = m_ptr->phase;
        }
        // Get emission contribution
        color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);

        BSDFSample sample_data = mat_ptr->sample(r_in, record, scattered);
        if (sample_data.type != BSDF_TYPE::DIFFUSE) { direct = false; }
        if (incoming_type != BSDF_TYPE::DIFFUSE || sample_data.type == BSDF_TYPE::TRANSMISSION) {
            accumulated_color += weight * color_from_emission;
        } else if (direct == false) { accumulated_color += weight * color_from_emission; } else {
            // To prevent double contribution of emission, only directly add if and only if:
            // => we are directly hitting the light, i.e (i==0)
            if (i == 0) { accumulated_color += weight * color_from_emission; }
        }

        // Direct Light Sampling
        if (direct && color_from_emission.length() == 0.0) {
            int N = (int)scene->physical_lights.size(); // amount of lights
            for (auto& light_ptr : scene->physical_lights) { // only accounts for physical lights currently
                // Create ray from hit point to the light
                point3 sampled_point = light_ptr->sample(record);
                vec3 light_dir = (sampled_point - record.pos).unit_vector(); // direction from hit point to the light
                ray light_ray = ray(record.pos, light_dir, 0.0);

                float distWithinMedium = (sampled_point - record.pos).length(); // distance to light

                // MultiIntersect to light to capture medium boundaries
                MediumRecord light_med_rec(record.pos);
                light_med_rec.mediums = med_rec.mediums;
                light_med_rec.highest_density_volume = med_rec.highest_density_volume;
                std::vector<int> ids;
                std::vector<float> tfars;
                // errors warning: overflow in conversion from 'float' to 'int' changes value from '+Inff' to '2147483647' [-Woverflow]
                // MultiIntersect(std::numeric_limits<float>::infinity(), light_ray, scene->rtc_scene, ids, tfars);
                int intersections_to_accept = 10;
                MultiIntersect(intersections_to_accept, light_ray, scene->rtc_scene, ids, tfars); // assume output ids.length() == tfars.length()

                // In this section, we fire trace each recorded intersection from pos -> light
                // It is hoped/assumed that we find it within 'intersections_to_accept' intersections or we find some obscurement
                bool non_medium_encountered = false;
                std::shared_ptr<Geometry> light_geomhit;
                int light_id;
                int light_tfar;
                for (size_t j = 0; j < ids.size(); j++) {
                    int id = ids[j];
                    float tfar = tfars[j];
                    light_geomhit = scene->geom_map[id];
                    if (!light_geomhit) { throw std::runtime_error("MultiIntersect returned some id that does not exist in geom_map"); }
                    std::shared_ptr<Volume> possible_volume_hit = std::dynamic_pointer_cast<Volume>(light_geomhit);
                    if (!possible_volume_hit) { // non volume encountered
                        if (light_geomhit == light_ptr) { // hit the light
                            light_id = id;
                            light_tfar = tfar;
                            light_med_rec.hitVolume(nullptr, light_ray.at(tfar));
                            break;
                        }
                        non_medium_encountered = true;
                        break;
                    } else { // one of the intersections was a volume
                        light_med_rec.hitVolume(possible_volume_hit->medium, light_ray.at(tfar));
                    }
                }
               
                if (!non_medium_encountered) { // if it is the light, we are not obscured from the light
                    // Store hit data of tracing the ray from here to the light
                    HitInfo light_record;
                    light_record = light_geomhit->getHitInfo(light_ray, light_ray.at(light_tfar), light_tfar, light_id);
                    
                    // Get the light's material
                    std::shared_ptr<material> light_mat_ptr = light_geomhit->materialById(light_id);

                    // Sample BSDF of hit point with incoming light
                    ray light_scattered;
                    
                    BSDFSample light_sample_data;
                    color att;
                    // We do NOT call the above line because it would sample a possibly different microfacet normal
                    // than what is already sampled previous to the Direct Light Sampling (for complex BSDFs that use microfacets)
                    // Both generate and pdf assume that, if a microfacet normal is needed, it is already defined. Thus, we use the previous
                    // and pass in the same HitInfo.
                    light_sample_data.bsdf_value = mat_ptr->generate(r_in, light_ray, record);
                    light_sample_data.pdf_value = mat_ptr->pdf(r_in, light_ray, record);
                    
                    // Find pdf for the light hit point
                    double light_pdf_value = light_ptr->pdf(light_record, light_ray);

                    // Find contribution of light using MIS power heuristic of light_pdf and sample pdf
                    double light_cos_theta = fabs(dot(record.normal, light_dir));
                    color light_contribution = weight * light_sample_data.bsdf_value * light_cos_theta
                            * MIS::power_heuristic<MIS::EVAL_WEIGHT>(light_pdf_value, light_sample_data.pdf_value) / light_pdf_value;
                    float transmittance_coeff = light_med_rec.transmittance;
                    light_contribution *= transmittance_coeff;
                    // Get emission of light
                    color light_Le = light_mat_ptr->emitted(light_record.u, light_record.v, light_record.pos);
                    accumulated_color += light_contribution * light_Le / N;
                }
            }
        }

        // Indirect ray contribution
        if (!sample_data.scatter) {
            return accumulated_color;
        }
        double cos_theta = fabs(dot(record.normal, (sample_data.scatter_direction)));
        if (!raymarched) {
            weight = weight * (sample_data.bsdf_value * cos_theta / sample_data.pdf_value);
        } else {
            weight = weight * (sample_data.bsdf_value / sample_data.pdf_value);
        }
        r_in = scattered;
        incoming_type = sample_data.type;
	}
    return accumulated_color;
}

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

    int sqrt_samples = int(sqrt(samples_per_pixel));


    for (int j=start_line; j>=start_line - (lines - 1); --j) {

        for (int i=0; i<image_width; ++i) {

            color pixel_color(0, 0, 0);

            for (int py = 0; py < sqrt_samples; ++py) {
                for (int px = 0; px < sqrt_samples; ++px) {
                    // Stratified sampling within the pixel
                    auto u = (i + (px + random_double()) / sqrt_samples) / (image_width - 1);
                    auto v = (j + (py + random_double()) / sqrt_samples) / (image_height - 1);
                    ray r = cam.get_ray(u, v);
                    pixel_color += trace_ray(r, scene_ptr, max_depth);
                }
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
