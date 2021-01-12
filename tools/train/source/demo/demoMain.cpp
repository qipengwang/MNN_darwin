//
//  demoMain.cpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "DemoUnit.hpp"
int main(int argc, const char* argv[]) {
//    printf("%s", __FILE_NAME__);
    if (argc < 2) {
        MNN_ERROR("Usage: ./runTrainDemo.out CASENAME [ARGS]\n");
        auto& list = DemoUnitSet::get()->list();
        MNN_PRINT("Valid Case: \n");

        for (auto iter : list) {
            MNN_PRINT("%s\n", iter.first.c_str());
        }
        return 0;
    }
    auto demo = DemoUnitSet::get()->search(argv[1]);
    if (nullptr == demo) {
        MNN_ERROR("Can't find demo %s\n", argv[1]);
        return 0;
    }
#ifdef __linux__
    printf("host is linux\n");
#else
    printf("host is MacOS\n");
#endif
    demo->run(argc - 1, argv + 1);
    return 0;
}
