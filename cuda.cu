#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct float10 {
  float x[10];
};

struct ptr4 {
  uchar3* v[4];
};

static __device__ __forceinline__ uchar3 belend(uchar3 a, uchar3 b, float w) {
  return make_uchar3(a.x * w + b.x * (1 - w), a.y * w + b.y * (1 - w),
                     a.z * w + b.z * (1 - w));
}

static __global__ void surround_kernel(const float10* table, int w, int h,
                                       ptr4 images, int iw, int ih,
                                       uchar3* output) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix >= w || iy >= h) return;

  int pos = iy * w + ix;
  float10 item = table[pos];
  int flag = item.x[0];
  float weight = item.x[1];

  if (flag == -1) return;
  if (flag < 4) {
    int x = item.x[2 + flag * 2 + 0];
    int y = item.x[2 + flag * 2 + 1];

    output[pos] = images.v[flag][y * iw + x];
  } else {
    const int idxs[][2] = {{2, 1}, {0, 3}, {0, 1}, {2, 3}};
    int a = idxs[flag - 4][0];
    int b = idxs[flag - 4][1];
    int ax = item.x[2 + a * 2 + 0];
    int ay = item.x[2 + a * 2 + 1];
    int bx = item.x[2 + b * 2 + 0];
    int by = item.x[2 + b * 2 + 1];
    output[pos] =
        belend(images.v[a][ay * iw + ax], images.v[b][by * iw + bx], weight);
  }
}

class Surrounder {
 public:
  virtual ~Surrounder() { destroy(); }

  bool load(const std::string& file, int w, int h, int numcam, int camw,
            int camh) {
    FILE* f = fopen(file.c_str(), "rb");
    if (f == nullptr) {
      printf("Failed to load table: %s\n", file.c_str());
      return false;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size != w * h * 10 * sizeof(float)) {
      printf("Invalid table file.\n");
      fclose(f);
      return false;
    }

    unsigned char* table_host = new unsigned char[size];
    fread(table_host, 1, size, f);
    fclose(f);

    w_ = w;
    h_ = h;
    camw_ = camw;
    camh_ = camh;
    output_.create(h_, w_, CV_8UC3);

    for (int i = 0; i < numcam; ++i) {
      unsigned char* device_ptr = nullptr;
      cudaMalloc(&device_ptr, camw * camh * 3 * sizeof(unsigned char));
      images_device_.push_back(device_ptr);
    }

    cudaMalloc(&output_view_, w_ * h_ * 3 * sizeof(unsigned char));
    cudaMalloc(&table_, size);
    cudaMemcpy(table_, table_host, size, cudaMemcpyHostToDevice);
    delete[] table_host;
    return true;
  }

  cv::Mat forward(const std::vector<cv::Mat>& images,
                  cudaStream_t stream = nullptr) {
    if (images.size() != images_device_.size()) {
      printf("Mismatched image size.\n");
      return cv::Mat();
    }

    for (int i = 0; i < images.size(); ++i) {
      auto& image = images[i];
      if (image.cols != camw_ || image.rows != camh_) {
        printf("Invalid image size: %d x %d\n", image.cols, image.rows);
        return cv::Mat();
      }

      cudaMemcpyAsync(images_device_[i], image.data,
                      image.cols * image.rows * 3 * sizeof(unsigned char),
                      cudaMemcpyHostToDevice, stream);
    }

    if (images.size() != 4) {
      printf("Unsupported image size.\n");
      return cv::Mat();
    }

    ptr4 images_ptr;
    memcpy(images_ptr.v, images_device_.data(), sizeof(images_device_[0]) * 4);
    dim3 block(32, 32);
    dim3 grid((w_ + block.x - 1) / block.x, (h_ + block.y - 1) / block.y);
    surround_kernel<<<grid, block, 0, stream>>>(
        table_, w_, h_, images_ptr, camw_, camh_, (uchar3*)output_view_);

    cudaMemcpyAsync(output_.data, output_view_,
                    output_.rows * output_.cols * 3 * sizeof(unsigned char),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return output_;
  }

 private:
  void destroy() {
    for (int i = 0; i < images_device_.size(); ++i) {
      cudaFree(images_device_[i]);
    }
    images_device_.clear();

    if (table_) {
      cudaFree(table_);
      table_ = nullptr;
    }

    if (output_view_) {
      cudaFree(output_view_);
      output_view_ = nullptr;
    }
  }

 private:
  std::vector<unsigned char*> images_device_;
  float10* table_ = nullptr;
  unsigned char* output_view_ = nullptr;
  cv::Mat output_;
  int w_ = 0;
  int h_ = 0;
  int camw_ = 0;
  int camh_ = 0;
};

int main() {
  Surrounder surround;
  if (!surround.load("surround_view.binary", 1200, 1600, 4, 960, 640)) {
    return -1;
  }

  const char* image_names[] = {"front", "left", "back", "right"};
  std::vector<cv::Mat> images;

  for (int i = 0; i < 4; ++i) {
    images.emplace_back(
        cv::imread(cv::format("images/%s.png", image_names[i])));
  }

  auto output = surround.forward(images);
  cv::imwrite("surround.jpg", output);
  printf("hello %d x %d\n", images[0].cols, images[0].rows);
  return 0;
}