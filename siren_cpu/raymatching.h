#pragma once

#include <image.h>
#include <camera_options.h>
#include <string>
#include <vector.h>
#include <sdf.h>
#include <initializer_list>
#include <ray.h>
#include <cmath>
#include <transformer.h>
#include <render_options.h>
#include <postprocessing.h>
#include <scene.h>
#include <omp.h>

#include <iostream>


bool OutOfBorder(const Vector& point) {
    return abs(point[0]) > 1.0 || abs(point[1]) > 1.0 || abs(point[2]) > 1.0;
}


Vector RayCast(const Scene& scene, Ray& view_ray, RenderOptions opt) {
    auto object = scene.GetUnionObject();
    double dist = object->ComputeSdf(view_ray.GetOrigin());
    bool is_inf = OutOfBorder(view_ray.GetOrigin());

    while (is_inf || dist > opt.eps) {

        if (is_inf) {
            return Vector();
        }
        
        double step = std::max(dist, opt.min_step);
        view_ray.ShiftOrigin(step);
        dist = object->ComputeSdf(view_ray.GetOrigin());
        is_inf =  OutOfBorder(view_ray.GetOrigin());
    }
    
    auto hitted_object = object->GetHittedObject(view_ray.GetOrigin());
    auto base_color = hitted_object->GetColor();

    auto light_dir = scene.GetLight().position - view_ray.GetOrigin();
    light_dir.Normalize();

    auto norm = EstimateNormal(hitted_object, view_ray.GetOrigin());
    double coeff = std::max(0.1, DotProduct(norm, light_dir));

    return coeff * scene.GetLight().intensity * base_color;
}


Image Render(const Scene& scene, const CameraOptions& camera_options, const RenderOptions& render_options) {
    Image img(camera_options.screen_width, camera_options.screen_height);
    Transformer transformer(camera_options);

    std::vector<std::vector<Vector>> color_map(img.Height(), std::vector<Vector>(img.Width()));
    auto begin = omp_get_wtime();
    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < img.Height(); ++i) {
        for (int j = 0; j < img.Width(); ++j) {
            Ray view_ray = transformer.MakeRay(i, j);
            color_map[i][j] = RayCast(scene, view_ray, render_options);
        }
    }
    std::cout << "CPU time " << omp_get_wtime() - begin << "\n";
    PostProc(img, color_map);
    return img;
}
