#include "detection_output.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

#include "bbox_util.h"
#include "bbox_util_gpu.h"

#include <iostream>

class Timer {
  public:
    void tic() {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc() {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
    }
    double t;
  private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

#define CUDA_CHECK(condition) \
    do { \
        cudaError_t error = condition; \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

DetectionOutputPlugin::DetectionOutputPlugin(const DetectionOutputParam& param)
{
    std::cout << "serialize constructor function" << std::endl;
    params = param;
    num_loc_classes_ = params.shareLocation ? 1 : params.numClasses;
    CHECK_GE(params.nmsThreshold, 0) << "nms threshold must be non negative.";
    eta_ = 1;
}

DetectionOutputPlugin::DetectionOutputPlugin(const void* buffer, size_t size)
{
    //std::cout << "deserialize constructor function" << std::endl;
    const char* d = static_cast<const char*>(buffer), *a = d;
    read<DetectionOutputParam> (d, params);
    read<int> (d, num_loc_classes_);
    read<int> (d, num_priors_);
    read<float> (d, eta_);
    
    assert(d == a + size);
}

DetectionOutputPlugin::~DetectionOutputPlugin() 
{
    std::cout << "descontructor function" << std::endl;
}

int DetectionOutputPlugin::getNbOutputs() const
{
    std::cout << "get number outputs function" << std::endl;
    return 1;
}

Dims DetectionOutputPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) 
{
    std::cout << "get output dimensions function" << std::endl;
    assert(index == 0 && nbInputDims == 5);

    // num_priors_ : prior box number
    num_priors_ = inputs[2].d[1] / 4;
    // inputs[2]->height() == inputs[0]->channels()
    CHECK_EQ(num_priors_ * num_loc_classes_ * 4, inputs[0].d[0])
        << "Number of priors must match number of location predictions.";
    // inputs[2]->height() / 4 * 21 == inputs[1]->channels()
    CHECK_EQ(num_priors_ * params.numClasses, inputs[1].d[0])
        << "Number of priors must match number of confidence predictions.";

    // output shape: {1 1 7}
    return Dims3(1, 1, 7);
}

bool DetectionOutputPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    std::cout << "supports format function" << std::endl;
    return type == DataType::kFLOAT and format == PluginFormat::kNCHW;
}

void DetectionOutputPlugin::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                                                int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
{
    std::cout << "#########################################" << std::endl;
    std::cout << "---------------configure with format function--------------------------" << std::endl;
    assert(type == DataType::kFLOAT and format == PluginFormat::kNCHW);
    assert(nbInputs == 5);



}

int DetectionOutputPlugin::initialize() 
{
    //std::cout << "initialize function" << std::endl;
    return 0;
}

void DetectionOutputPlugin::terminate() 
{
    std::cout << "terminate function" << std::endl;
}

size_t DetectionOutputPlugin::getWorkspaceSize(int maxBatchSize) const
{
    std::cout << "get workspace size function" << std::endl;
    return 0;
}

size_t DetectionOutputPlugin::getSerializationSize()
{
    std::cout << "get serialization size function" << std::endl;
    return sizeof(DetectionOutputParam) + sizeof(int) + sizeof(int) + sizeof(float);
}

//odm_loc:          1 25500 1 1
//odm_conf_flatten: 1 133875 1 1
//arm_priorbox:     1 2 25500 1   prior box number: 25500 / 4 = 6375 = 40 x 40 x 3 + 20 x 20 x 3 + 10 x 10 x 3 + 5 x 5 x 3
//arm_conf_flatten: 1 12750 1 1
//arm_loc:          1 25500 1 1  
//top shape:        1 1 1 7 -> 1 1 num_kept 7

int DetectionOutputPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    //std::cout << "enqueue function" << std::endl;
    const float* loc_data = (const float*)inputs[0]; // get gpu pointer
    const float* prior_data = (const float*)inputs[2]; 
    const float* arm_loc_data = NULL;
    const int num = 1;
    arm_loc_data = (float*)inputs[4];


	int conf_size = num_priors_ * params.numClasses;
	int loc_size = num_priors_ * 4;
    
    // 1.Decode predictions
    // note1: arm_loc + prior box -> bbox_data 
    // note2: bbox_data + odm_loc + prior box -> bbox_data
    // note3: box1[x1, y1, x2, y2], box2[x1, y1, x2, y2] ... box6375[x1, y1, x2, y2]
    float* bbox_data = NULL; // allocate GPU memory
    CUDA_CHECK(cudaMalloc(&bbox_data, loc_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bbox_data, 0, loc_size * sizeof(float)));
    const int loc_count = loc_size; // 25500
    const bool clip_bbox = false;

    //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //std::cout << "num priors: " << num_priors_ << std::endl;

    CasRegDecodeBBoxesGPU<float>(loc_count, loc_data, prior_data, params.codeType,
                                params.varianceEncodedInTarget, num_priors_, params.shareLocation,
                                num_loc_classes_, params.backgroundLabelId, clip_bbox, bbox_data, arm_loc_data);
    


    // move decoded location predictions from GPU to CPU
    float* bbox_cpu_data = NULL; // allocate CPU memory
    CUDA_CHECK(cudaMallocHost(&bbox_cpu_data, loc_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(bbox_cpu_data, (const void*)bbox_data, loc_size * sizeof(float), cudaMemcpyDeviceToHost));
    


    // 2.Retrieve all confidences
    float* conf_permute_data = NULL; // allocate GPU memory
    CUDA_CHECK(cudaMalloc(&conf_permute_data, conf_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(conf_permute_data, 0, conf_size * sizeof(float)));

    const float* odm_conf = (const float*)inputs[1]; // get gpu pointer  
    const float* arm_conf = (const float*)inputs[3]; 

    //float* tmp = new float[100];
    //cudaMemcpy(tmp, odm_conf, 100 * sizeof(float), cudaMemcpyDeviceToHost);

    //std::cout << "___________________________" << std::endl;
    //for (int i = 0; i < 100; i++)
    //    std::cout << tmp[i] << " ";
    //std::cout << std::endl;

    // note1: use arm_conf -> judge, use odm_conf -> assign value
    // note2: format: class0[6375], class1[6375], class2[6375] ... class21[6375]
    OSPermuteDataGPU<float>(conf_size, odm_conf, arm_conf, params.numClasses, num_priors_, 1, conf_permute_data, params.objectnessScore);

    // move confidences from GPU to CPU
    float* conf_cpu_data = NULL;
    CUDA_CHECK(cudaMallocHost(&conf_cpu_data, conf_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(conf_cpu_data, (const void*)conf_permute_data, conf_size * sizeof(float), cudaMemcpyDeviceToHost));

    //std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl; 
    //for (int i = 0; i < 100; i++)
    //    std::cout << conf_cpu_data[i] << " ";
    //std::cout << std::endl;

    // 3.post-process detection results
    int num_kept = 0;
    vector<map<int, vector<int> > > all_indices; // vector<...> usage situation: batchsize > 1
                                                 // map<int, vector<int> > -> (label, multiple idxs)
    
	std::cout << "classes: " << params.numClasses << std::endl;
	std::cout << "num priors: " << num_priors_ << std::endl;

    // num -> batchsize
    for (int i = 0; i < num; ++i) {
        // key -> class index
        // value -> stroe box id (nms) specify class index 1, 2 ... 20
        map<int, vector<int> > indices;
        int num_det = 0;
		const int conf_idx = 0;
		int bbox_idx = 0;
        /*for (int c = 0; c < params.numClasses; ++c) {
            if (c == params.backgroundLabelId) {
                // Ignore background class
                continue;
            }
			// params.nmsThreshold
            const float* cur_conf_data = conf_cpu_data + c * num_priors_;
            const float* cur_bbox_data = bbox_cpu_data;
            
            Timer timer;
            timer.tic();
			//std::cout << "debug: " << *cur_conf_data << " " << *cur_bbox_data << std::endl;
			std::cout << "topK: " << params.topK << std::endl;
            ApplyNMSFast<float>(cur_bbox_data, cur_conf_data, num_priors_,
                params.confidenceThreshold, 0.35, eta_, params.topK, &(indices[c]));
            timer.toc();       
            double msTime = timer.t;
            //std::cout << "nms time: " << msTime << "ms" << std::endl;  
            num_det += indices[c].size();
            std::cout << "class index: " << c << " bbox size: " << indices[c].size() << std::endl;
        }*/

		
		const float* cur_conf_data = conf_cpu_data + num_priors_;
		const float* cur_bbox_data = bbox_cpu_data;
		//std::cout << "debug " << std::endl;
		ApplyNMSFast<float>(cur_bbox_data, cur_conf_data, num_priors_ * (params.numClasses - 1), params.numClasses, 
				params.confidenceThreshold, 0.3, eta_, params.topK, &indices);
		//std::cout << "debug debug " << std::endl;
		num_det += indices[1].size();
		num_det += indices[2].size();
		std::cout << "num_det: " << num_det << " = " << indices[1].size() << " + " << indices[2].size() << std::endl;
				
        /*
		// lampson
		const float* cur_conf_data = NULL;
        const float* cur_bbox_data = NULL;
		int start_idx = 0;
        for (int c = 0; c < params.numClasses; ++c) {
            if (c == params.backgroundLabelId) {
                // Ignore background class
                continue;
            }   
			start_idx = c;

            cur_conf_data = conf_cpu_data + conf_idx + c * num_priors_;
            cur_bbox_data = bbox_cpu_data + bbox_idx;
			break;
        }   
  
        ApplyNMSFast<float>(cur_bbox_data, cur_conf_data, (params.numClasses - start_idx)*num_priors_,
            params.confidenceThreshold, 0.3, eta_, params.topK, &(indices[start_idx]));
        cur_conf_data = NULL;
        cur_bbox_data = NULL;
        num_det += indices[start_idx].size();*/
		      
        // each class (topK 1000 -> nms) -> 21 classes num_det -> 21 classes keeptopk 500
        // keeptopk: sort keeptopk according to score 
        //std::cout << "num_det value: " << num_det << std::endl;
		//std::cout << "keepTopK param: " << params.keepTopK << std::endl;
		
        if (params.keepTopK > -1 && num_det > params.keepTopK) {
            vector<pair<float, pair<int, int> > > score_index_pairs; // (score, (label, id))
            for (map<int, vector<int> >::iterator it = indices.begin(); it != indices.end(); ++it) {
                int label = it->first;
                const vector<int>& label_indices = it->second;
                for (int j = 0; j < label_indices.size(); ++j) {
                    int idx = label_indices[j];
                    float score = conf_cpu_data[conf_idx + label * num_priors_ + idx];
                    score_index_pairs.push_back(std::make_pair(score, std::make_pair(label, idx)));
                }
            } 
            // keeptopk results per image
            std::sort(score_index_pairs.begin(), score_index_pairs.end(), SortScorePairDescend<pair<int, int> >);
            score_index_pairs.resize(params.keepTopK);
            // store the new indices
            map<int, vector<int> > new_indices; // (label, multiple idxs)
            for (int j = 0; j < score_index_pairs.size(); ++j) {
                int label = score_index_pairs[j].second.first;
                int idx = score_index_pairs[j].second.second;
                new_indices[label].push_back(idx);
            }
            all_indices.push_back(new_indices);
            num_kept += params.keepTopK;
        } else {
			std::cout << "not use keepTopK" << std::endl;
            all_indices.push_back(indices);
            num_kept += num_det;
        }
    }
    
	//std::cout << "num kept value: " << num_kept << std::endl;

    vector<int> top_shape(2, 1);
    top_shape.push_back(num_kept);
    top_shape.push_back(7);
    
    //  top_data -> allocate CPU memory
    float* top_data = NULL;
    
    // allocate CPU memory;
    if (num_kept == 0) {
        std::cout << "couldn't find any detections"<<std::endl;
        top_shape[2] = num;
        // allocate memory  use fake top_shape
        CUDA_CHECK(cudaMallocHost(&top_data, 1 * 1 * 1 * 7 * sizeof(float)));
        CUDA_CHECK(cudaMemset(top_data, 0, 1 * 1 * 1 * 7 * sizeof(float)));

    } else {
        // allocate memory according to top_shape
        CUDA_CHECK(cudaMallocHost(&top_data, 1 * 1 * (num_kept+1) * 7 * sizeof(float)));
        CUDA_CHECK(cudaMemset(top_data, 0, 1 * 1 * (num_kept+1) * 7 * sizeof(float)));
    }
    
    //std::cout << "all indices size: " << all_indices.size() << std::endl;

    // top_data assign value, its format [image_id, class, confidence, x1, y1, x2, y2]


    int count = 0;
    for (int i = 0; i < num; ++i) {
        const int conf_idx = 0;
        int bbox_idx = 0;
        for (map<int, vector<int> >::iterator it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
			// map<int, vector<int> > 
            int label = it->first; // label start from 1
			std::cout << "label: " << label << std::endl;
            vector<int>& indices = it->second;
			std::cout << "verify size: " << indices.size() << std::endl;
			
			float* cur_conf_data = NULL;
			float* cur_bbox_data = NULL;
			
			cur_conf_data = conf_cpu_data + label * num_priors_;
            cur_bbox_data = bbox_cpu_data;


            for (int j = 0; j < indices.size(); ++j) {
                int idx = indices[j];
				if (idx > num_priors_) {
					idx = idx % num_priors_;
				}
                top_data[count * 7] = i+1;
                top_data[count * 7 + 1] = label;
                top_data[count * 7 + 2] = cur_conf_data[idx];
				//std::cout << cur_conf_data[idx] << std::endl;
                for (int k = 0; k < 4; ++k) {
                    top_data[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
                }
                ++count;
            }
        }        
    } 

	//std::cout << "count: " << count << std::endl;

    //std::cout << "last last last last" << std::endl;
    /*for (int i = 0; i < num_kept + 10 ; i++) {
        std::cout << top_data[7*i] << " " << top_data[7*i+1] << " " << top_data[7*i+2] << " "
				  << top_data[7*i+3] << " " << top_data[7*i+4] << " " << top_data[7*i+5] << " "
				  << top_data[7*i+6];
		std::cout << std::endl;
	}*/
    
    
    //outputs[0] = top_data;
    //outputs[0] = new float[10];
    //std::cout<<"output[0] pointer : " <<outputs[0] << std::endl;
    //float* a = new float[?];
    //a[0] = 4;
    //a[1] = 1;
    //cudaMemcpy(outputs[0], a, ? * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout<<"aaaaaaaaaaaaaaa output : " << *((float*)outputs[0]) << std::endl;
    //cudaDeviceSynchronize();
    // free memory
    
	
    CUDA_CHECK(cudaMemcpy(outputs[0], top_data, 7 * num_kept * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(bbox_data));
    CUDA_CHECK(cudaFree(conf_permute_data));
    CUDA_CHECK(cudaFreeHost(bbox_cpu_data));    
    CUDA_CHECK(cudaFreeHost(conf_cpu_data));
    CUDA_CHECK(cudaFreeHost(top_data));
    return 0;
}

void DetectionOutputPlugin::serialize(void* buffer)
{
    std::cout << "serialize function" << std::endl;
    char* d = static_cast<char*>(buffer), *a = d;
    std::cout << "num loc classes: " << num_loc_classes_ << std::endl;
    std::cout << "num_priors_: " << num_priors_ << std::endl;
    std::cout << "eta_: " << eta_ << std::endl;
    write<DetectionOutputParam> (d, params);
    write<int> (d, num_loc_classes_);
    write<int> (d, num_priors_);
    write<float> (d, eta_);
    assert(d == a + getSerializationSize());
}
