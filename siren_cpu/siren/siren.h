#pragma once

#include <vector>
#include <fstream>
#include <nn_layer.h>
#include <linear_layer.h>
#include <activation_layer.h>


std::vector<float> readBinaryArray(std::string fileName) {
    std::vector<float> ret;
    float f;
    std::ifstream fin(fileName, std::ios::binary);
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        ret.push_back(f);
    }
    return ret;
 }


class Siren {
private:
	std::vector<NNLayer*> layers;

public:
	Siren(const std::vector<std::pair<Shape, std::string>> &arch, std::string weights_path);
	~Siren();

	Matrix forward(const Matrix& X) const;

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};


Siren::Siren(const std::vector<std::pair<Shape, std::string>> &arch, std::string weights_path) {
    auto weights = readBinaryArray(weights_path);
    size_t pos = 0;

    for (const auto& layer : arch) {
        if (layer.second == "fc") {
            auto fc = new LinearLayer(layer.first);
            for (int i = 0; i < layer.first.x * layer.first.y; ++i) {
                fc->W[i] = weights[pos++];
            }

            for (int i = 0; i < layer.first.y; ++i) {
                fc->b[i] = weights[pos++];
            }
            this->addLayer(fc);
        } else {
            this->addLayer(new SinLayer());
        }
    }
}


Siren::~Siren() {
	for (auto layer : layers) {
		delete layer;
	}
}


void Siren::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}


Matrix Siren::forward(const Matrix& X) const {
	Matrix Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}
	return Z;
}


std::vector<NNLayer*> Siren::getLayers() const {
	return layers;
}
