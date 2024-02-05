#ifndef TIMELINE_H
#define TIMELINE_H

#include <vector>
#include "vec3.h"


struct TimePosition {
    double time;
    vec3 position;
    int transition;
};

class timeline {
    public:
        timeline(std::vector<TimePosition> tp) : motion(tp) {};
        vec3 interpolate_position(double time) const {
            TimePosition before = motion.front();
            TimePosition after = motion.back();

            for (const auto& tp : motion) {
                if (tp.time < time && tp.time > before.time) {
                    before = tp;
                } else if (tp.time >= time && tp.time < after.time) {
                    after = tp;
                }
            }

            int Transition = before.transition;
            // Linear
            //if (Transition == 0) {
            double t = (time - before.time) / (after.time - before.time);
            return before.position * (1 - t) + after.position * t;
            //}
        }
    public:
        std::vector<TimePosition> motion;
};

#endif