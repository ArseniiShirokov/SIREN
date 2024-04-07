#pragma once

#include <sdf.h>
#include <vector.h>
#include <light.h>

#include <vector>
#include <string>
#include <initializer_list>
#include <union_sdf.h>


class Scene {
public:
    Scene(std::vector<SDF*> objects, Light light)
        : union_object_(std::move(objects)),
          light_(light) {
    }

    const UnionSDF* GetUnionObject() const {
        return &union_object_;
    }

    const Light& GetLight() const {
        return light_;
    }


private:
    UnionSDF union_object_;
    Light light_;
};
