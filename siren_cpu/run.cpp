#include <catch.hpp>

#include <cmath>
#include <string>
#include <optional>

#include <sphere.h>
#include <scene.h>

#include <camera_options.h>
#include <image.h>
#include <raymatching.h>
#include <render_options.h>
#include <scene.h>
#include <light.h>
#include <neural_sdf.h>
#include <matrix.h>
#include <linear_layer.h>
#include <activation_layer.h>
#include <siren.h>

#include <vector>
#include <iostream>


int artifact_index = 0;
const std::string kArtifactsDir = BASE_DIR;


void SaveImage(const std::string& result_filename, Image& image) {
    image.Write(kArtifactsDir + "/results/" + result_filename);
}


void CompareMLPOutputs(Siren& mlp, const std::string& test_file) {
    float f;
    int num;
    std::ifstream fin(test_file, std::ios::binary);

    std::vector<float> distances;
    std::vector<float> arr;

    fin.read(reinterpret_cast<char*>(&num), sizeof(num));

    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        arr.push_back(f);
    }

    size_t pos = 0;
    Shape shape(1, 3);
    Matrix point(shape);


    for (size_t i = 0; i < num; ++i) {
       point[0] = arr[pos++];
       point[1] = arr[pos++];
       point[2] = arr[pos++];

       auto dist = mlp.forward(point)[0];
       distances.push_back(dist);
    }

    for (size_t i = 0; i < num; ++i) {
        REQUIRE(abs(arr[pos++] - distances[i]) < 1e-5);
    }
}


TEST_CASE("sdf1_test", "[siren]") {
    std::vector<std::pair<Shape, std::string>> arch;
    arch.push_back(std::make_pair(Shape(3, 64), "fc"));
    arch.push_back(std::make_pair(Shape(64, 64), "sin"));
    arch.push_back(std::make_pair(Shape(64, 64), "fc"));
    arch.push_back(std::make_pair(Shape(64, 64), "sin"));
    arch.push_back(std::make_pair(Shape(64, 64), "fc"));
    arch.push_back(std::make_pair(Shape(64, 64), "sin"));
    arch.push_back(std::make_pair(Shape(64, 1), "fc"));
    Siren mlp(arch, kArtifactsDir + "/files/sdf1_weights.bin");

    CompareMLPOutputs(mlp, kArtifactsDir + "/files/sdf1_test.bin");
}


TEST_CASE("sdf2_test", "[siren]") {
    std::vector<std::pair<Shape, std::string>> arch;
    arch.push_back(std::make_pair(Shape(3, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));

    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));

    arch.push_back(std::make_pair(Shape(256, 1), "fc"));
    Siren mlp(arch, kArtifactsDir + "/files/sdf2_weights.bin");
    CompareMLPOutputs(mlp, kArtifactsDir + "/files/sdf2_test.bin");
}


TEST_CASE("render_sdf1", "[siren]") {
    std::cout << "Render sdf1...\n";

    std::vector<std::pair<Shape, std::string>> arch;
    arch.push_back(std::make_pair(Shape(3, 64), "fc"));
    arch.push_back(std::make_pair(Shape(64, 64), "sin"));
    arch.push_back(std::make_pair(Shape(64, 64), "fc"));
    arch.push_back(std::make_pair(Shape(64, 64), "sin"));
    arch.push_back(std::make_pair(Shape(64, 64), "fc"));
    arch.push_back(std::make_pair(Shape(64, 64), "sin"));
    arch.push_back(std::make_pair(Shape(64, 1), "fc"));
    Siren* mlp = new Siren(arch, kArtifactsDir + "/files/sdf1_weights.bin");


    CameraOptions camera_opts(256, 256);
    camera_opts.look_from = {0, 0, -0.99};
    camera_opts.look_to = {0, 0, 0};
    RenderOptions render_opts;

    auto object = new NeuralSDF(mlp, Vector{0.5, 1, 1});
    
    std::vector<SDF*> objects;
    objects.push_back(object);

    auto position = Vector{-4, 5, -1.0};
    auto intensity = Vector{0.1, 0.1, 0.1};
    auto light = Light(position, intensity);

    auto scene = Scene(objects, light);
    auto image = Render(scene, camera_opts, render_opts);
    SaveImage("sdf1_cpu.png", image);
}


TEST_CASE("render_sdf2", "[siren]") {
    std::cout << "Render sdf2...\n";

    std::vector<std::pair<Shape, std::string>> arch;
    arch.push_back(std::make_pair(Shape(3, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 256), "fc"));
    arch.push_back(std::make_pair(Shape(256, 256), "sin"));
    arch.push_back(std::make_pair(Shape(256, 1), "fc"));
    Siren* mlp = new Siren(arch, kArtifactsDir + "/files/sdf2_weights.bin");


    CameraOptions camera_opts(256, 256);
    camera_opts.look_from = {0, 0, -0.99};
    camera_opts.look_to = {0, 0, 0};
    RenderOptions render_opts;

    auto object = new NeuralSDF(mlp, Vector{0.5, 1, 1});
    
    std::vector<SDF*> objects;
    objects.push_back(object);

    auto position = Vector{-4, 5, -1.0};
    auto intensity = Vector{0.1, 0.1, 0.1};
    auto light = Light(position, intensity);

    auto scene = Scene(objects, light);
    auto image = Render(scene, camera_opts, render_opts);
    SaveImage("sdf2_cpu.png", image);
}
