//
//  MobilenetV2Utils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#define DEBUG
#include "MobilenetV2Utils.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include <MNN/expr/NN.hpp>
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN_generated.h>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "ImageDataset.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

std::string numberToString(int index) {
    char s[10];
    snprintf(s, 10, "%d", index);
    return std::string(s);
}

void MobilenetV2Utils::train(std::shared_ptr<Module> model, const int numClasses, const int addToLabel,
                                std::string trainImagesFolder, std::string trainImagesTxt,
                                std::string testImagesFolder, std::string testImagesTxt,
                                const int trainQuantDelayEpoch, const int quantBits) {
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 2);
    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setMomentum(0.9f);
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(0.00004f);

    auto converImagesToFormat  = CV::RGB;
    int resizeHeight           = 224;
    int resizeWidth            = 224;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
    std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
    bool centerOrRandomCrop = false; // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means,cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto trainDataset = ImageDataset::create(trainImagesFolder, trainImagesTxt, datasetConfig.get(), readAllImagesToMemory);
    auto testDataset = ImageDataset::create(testImagesFolder, testImagesTxt, datasetConfig.get(), readAllImagesToMemory);

    const int trainBatchSize = 4;
    const int trainNumWorkers = 1;
    const int testBatchSize = 1;
    const int testNumWorkers = 0;

    auto trainDataLoader = trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);
    auto testDataLoader = testDataset.createLoader(testBatchSize, true, false, testNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();
    const int testIterations = testDataLoader->iterNumber();

    // const int usedSize = 1000;
    // const int testIterations = usedSize / testBatchSize;

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            trainDataLoader->reset();
            model->setIsTraining(true);
            // turn float model to quantize-aware-training model after a delay
            if (epoch == trainQuantDelayEpoch) {
                // turn model to train quant model
                std::static_pointer_cast<PipelineModule>(model)->toTrainQuant(quantBits);
            }
            for (int i = 0; i < trainIterations; i++) {
                AUTOTIME;
                printf("solver->swapable().size() = %lu,\tsolver->trainable().size() = %lu\n", solver->swapable().size(),solver->trainable().size());
                for(auto p: solver->swapable()) {
                    p->clearInput2Expr();
                }
                auto trainData  = trainDataLoader->next();
                auto example    = trainData[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                                  _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));
                printf("call model.forward to build forward propagation graph\n");
                auto predict = model->forward(_Convert(example.first[0], NC4HW4));
                auto loss    = _CrossEntropy(predict, newTarget);
//                loss->setName("loss");
//                Variable::save({loss}, "swap/loss.mobilenetv2.mnn");
//                predict->setName("prob");
//                Variable::save({predict}, "swap/prob.mobilenetv2.mnn");
                // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                float rate = 1e-5;
                solver->setLearningRate(rate);
                if (solver->currentStep() % 5 == 0) {
                    std::cout << "train iteration: " << solver->currentStep();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate << std::endl;
                }
//                auto executeOrder = Variable::getExecuteOrder({loss});
//                int index=0;
//                for (auto iter = executeOrder.rbegin(); iter != executeOrder.rend(); iter++) {
//                    auto expr = *iter; // expr object
//                    auto& name = expr->name();
//                    if(name.empty()) {
//                        MNN_PRINT("empty, set default: %s\t", (EnumNameOpType(OpType_Const) + numberToString(++index)).c_str());
////                         continue;
//                        int outSize=expr->outputSize();
//                        MNN_PRINT("outputSize = %d:\t", outSize);
////                        continue;
//                        for(int p=0; p < outSize; p++){
//                            if (expr->outputName(p).empty())
//                                MNN_PRINT("#\t");
//                            else
//                                MNN_PRINT("%s\t", expr->outputName(p).c_str());
//                        }MNN_PRINT("\n");
//                    } else {
//                        MNN_PRINT("name = %s\n", name.c_str());
//                    }
//                }
//                return;
                solver->step(loss);
                MNN_PRINT("finish one iteration\n");
                return;
            }
        }

        int correct = 0;
        int sampleCount = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        exe->gc(Executor::PART);

        AUTOTIME;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            auto predict    = model->forward(_Convert(example.first[0], NC4HW4));
            predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
            auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
            sampleCount += label->getInfo()->size;
            auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));
            correct += accu->readMap<int32_t>()[0];

            if ((i + 1) % 10 == 0) {
                std::cout << "test iteration: " << (i + 1) << " ";
                std::cout << "acc: " << correct << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%";
                std::cout << std::endl;
            }
        }
        auto accu = (float)correct / testDataLoader->size();
        // auto accu = (float)correct / usedSize;
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;

        {
            auto forwardInput = _Input({1, 3, resizeHeight, resizeWidth}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            Transformer::turnModelToInfer()->onExecute({predict});
            predict->setName("prob");
            Variable::save({predict}, "temp.mobilenetv2.mnn");
        }

        exe->dumpProfile();
    }
}
