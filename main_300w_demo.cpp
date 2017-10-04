#include <caffe/caffe.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

//#define REALIGN_PTS	// uncomment to realign points with [1:17 27:31 38:42 18:26 32:37 43:68] to measure accuracy

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

void WrapInputLayer(shared_ptr<Net<double> > net_, std::vector<cv::Mat>* input_channels);
void Preprocess(shared_ptr<Net<double> > net_, const cv::Mat& img, std::vector<cv::Mat>* input_channels);
template <typename Dtype>
string OutputOfBlobByName(shared_ptr<Net<Dtype> > net_, const string& blob_name);
std::vector<string> TextRead(string filename);

int main(int argc, char** argv) {
  if (argc != 5) {
	  std::cerr << "Usage: " << argv[0]
			 << " deploy.prototxt network.caffemodel"
			 << " imglist.txt test_mode" << std::endl;
	  return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  // image list
  string imglist = argv[3];	// "300w_img_list.txt";
  // model config
  int IMG_DIM = 448;
  string model_file = argv[2];	// "Models/300W/network_300W_parts.caffemodel";
  string model_def_file = argv[1];	// "Models/300W/network_300W_parts.prototxt";

  // use gpu
  int gpuDevice = 0;
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(gpuDevice);

  // Initialize a network
  Phase phase = TEST;
  shared_ptr<Net<double> > net(new caffe::Net<double>(model_def_file, phase));
  net->CopyTrainedLayersFrom(model_file);

  std::vector<string> img_list = TextRead(imglist);
  std::vector<string> rct_list, pts_list;
  for (size_t i=0; i< img_list.size(); i++) {
	  img_list[i] = img_list[i].substr(0, img_list[i].size()-1);	// '\r' character is at the end
	  string img_line = img_list[i].substr(0, img_list[i].size()-3);
	  rct_list.push_back(img_line + "rct");
	  pts_list.push_back(img_line + "pts");
  }

  /* Face Detection */
  cv::CascadeClassifier cascade;
  cascade.load( "OpenCV_Cascades/haarcascade_frontalface_default.xml" ) ;

  // show result
  int vis_result = 1;

  size_t num = img_list.size();
  std::vector<cv::Mat> detected_points(num, cv::Mat(68, 2, CV_64FC1, 0.0));
  std::vector<cv::Mat> ground_truth(num, cv::Mat(68, 2, CV_64FC1, 0.0));
  for (size_t j = 0; j < num; j++) {

	  std::cout << j+1 << '/' << num << std::endl;
	  cv::Mat src_img = cv::imread(img_list[j], -1);

	  // expand face roi
	  std::vector<int> rct(4), src_rct(4);
	  if (1) {
		  double sh_scale = 0.1;
		  std::vector<string> rct_lines = TextRead(rct_list[j]);
		  if (!rct_lines.empty()) {
			  std::stringstream ss_rct_line(rct_lines[0]);	// only one line in this file
			  ss_rct_line >> rct[0] >> rct[1] >> rct[2] >> rct[3];
		  }
		  else {
			  std::cerr << "Warning: .rct file not found! Using OpenCV's face detector..." << std::endl;
			  std::vector<cv::Rect> faces;
			  cascade.detectMultiScale( src_img, faces, 1.1,
			                            2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );
			  if (faces.empty()) {
				  std::cerr << "Warning:    OpenCV could not find a face! Skipping to the next image" << std::endl;
				  break;
			  }
			  // use the largest face found
			  int largest_width = 0;
			  int largest_width_idx = 0;
			  size_t f = 0;
			  for (; f<faces.size(); f++) {
				  if (faces[f].width > largest_width) {
					  largest_width = faces[f].width;
					  largest_width_idx = f;
				  }
			  }
			  rct[0] = faces[largest_width_idx].x;
			  rct[1] = faces[largest_width_idx].y;
			  rct[2] = faces[largest_width_idx].x + faces[largest_width_idx].width;
			  rct[3] = faces[largest_width_idx].y + faces[largest_width_idx].height;
		  }
		  src_rct = rct;

		  double row = src_img.rows;
		  double col = src_img.cols;
		  double w = rct[2] - rct[0];
		  double h = rct[3] - rct[1];
		  std::vector<double> rct_float(4);
		  rct_float[0] = rct[0] - sh_scale*w;
		  rct_float[1] = rct[1] - sh_scale*h;
		  rct_float[2] = rct[2] + sh_scale*w;
		  rct_float[3] = rct[3] + sh_scale*h;

		  for (size_t r = 0; r < 4; r++) {	// TODO: try threaded loop here
			  rct[r] = round(rct_float[r]);
		  }
		  if (rct[0] <= 0) {
			  rct[0] = 1;
		  }
		  if (rct[1] <= 0) {
			  rct[1] = 1;
		  }
		  if (rct[2] > col) {
			  rct[2] = col;
		  }
		  if (rct[3] > row) {
			  rct[3] = row;
		  }
		  std::vector<string> pts_lines = TextRead(pts_list[j]);
		  for (size_t g=0; g<pts_lines.size(); g++) {
			  std::stringstream ss_pts_line(pts_lines[g]);
			  std::vector<double> g_pts(2);
			  ss_pts_line >> g_pts[0] >> g_pts[1];
			  ground_truth[j].at<double>(g, 0) = g_pts[0];
			  ground_truth[j].at<double>(g, 1) = g_pts[1];
		  }
	  }

	  /* */
	  cv::Mat im = src_img(cv::Range(rct[1], rct[3]), cv::Range(rct[0], rct[2])).clone();
	  cv::resize(im, im, cv::Size(IMG_DIM, IMG_DIM));	// TODO: try different interpolation methods

	  //TODO: try permute channels (currently passing as it is, whatever it is)
	  //%im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
	  //%images(:,:,:, 1) = permute(im_data, [2, 1, 3]); // passing GBR

	  std::vector<cv::Mat> input_channels;
	  WrapInputLayer(net, &input_channels);
	  Preprocess(net, im, &input_channels);

	  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
	  net->Forward();
	  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();

//	  std::vector<string> bnames = net->blob_names();
//	  std::cout << OutputOfBlobByName(net, "downsample_data") << std::endl;
	  std::cout << OutputOfBlobByName(net, "theta") << std::endl;

	  /* Copy the output layer to a std::vector */
	  Blob<double>* output_layer = net->output_blobs()[0];
	  const double* begin = output_layer->cpu_data();
	  const double* end = begin + output_layer->channels();
	  std::vector<double> output = std::vector<double>(begin, end);

	  /* */
	  std::vector<double> fea(2);
	  cv::Mat pre_pts1(27, 2, CV_64FC1, 0.0);
	  cv::Mat pre_pts2(41, 2, CV_64FC1, 0.0);
	  size_t prept1 = 0;
	  size_t prept2 = 0;
	  double scale_x = (rct[2] - rct[0] + 1.0) / IMG_DIM;
	  double scale_y = (rct[3] - rct[1] + 1.0) / IMG_DIM;
	  for (size_t p=0; p<output.size(); p+=2) {
		  fea[0] = (output[p] + 1) * IMG_DIM / 2;
		  fea[1] = (output[p+1] + 1) * IMG_DIM / 2;
		  fea[0] = fea[0] * scale_x + rct[0];
		  fea[1] = fea[1] * scale_y + rct[1];

//		  std::vector<double> rea(2);
//		  rea[0] = ground_truth[j].at<double>(p/2, 0);
//		  rea[1] = ground_truth[j].at<double>(p/2, 1);//
//		  std::cout << "Err" << p << ": " << std::sqrt( std::pow(rea[0]-fea[0], 2) + std::pow(rea[1]-fea[1], 2) ) << std::endl;
#ifndef REALIGN_PTS
		  detected_points[j].at<double>(p/2, 0) = fea[0];
		  detected_points[j].at<double>(p/2, 1) = fea[1];
	  }
#else

		  if ((0 <= p && p <= 32)
				  || (52 <= p && p <= 60)
				  || (74 <= p && p <= 82)) {	// %[1:17 27:31 38:42]
			  pre_pts1.at<double>(prept1, 0) = fea[0];
			  pre_pts1.at<double>(prept1, 1) = fea[1];
			  prept1++;
		  }
		  else {	// %[18:26 32:37 43:68]
			  pre_pts2.at<double>(prept2, 0) = fea[0];
			  pre_pts2.at<double>(prept2, 1) = fea[1];
			  prept2++;
		  }
	  }
	  for (size_t d=0; d<27; d++) {
		  detected_points[j].at<double>(d, 0) = pre_pts1.at<double>(d,0);
		  detected_points[j].at<double>(d, 1) = pre_pts1.at<double>(d,1);
	  }
	  for (size_t d=27; d<68; d++) {
		  detected_points[j].at<double>(d, 0) = pre_pts2.at<double>(d-27,0);
		  detected_points[j].at<double>(d, 1) = pre_pts2.at<double>(d-27,1);
	  }
#endif
	  
	  /* show results */
	  if (vis_result) {
		  double w = src_rct[2]-src_rct[0]+1.0;
		  double h = src_rct[3]-src_rct[1]+1.0;
		  int l_w = 2;	//std::max((int)round(w/100), 3);
		  int p_w = 2; 	//std::max(5, (int)round(w/15));

		  cv::Mat src_img2 = src_img;
		  cv::rectangle(src_img2, cv::Rect(src_rct[0], src_rct[1], w, h), cv::Scalar(0), l_w);
		  // draw pre pts
		  for (size_t c=0; c<68; c++) {
			  cv::circle(src_img2, cv::Point(detected_points[j].at<double>(c, 0), detected_points[j].at<double>(c, 1)), p_w+1, cv::Scalar(0,255,0), -1);
			  cv::circle(src_img2, cv::Point(ground_truth[j].at<double>(c, 0), ground_truth[j].at<double>(c, 1)), p_w, cv::Scalar(0,0,255), -1);
		  }

		  cv::namedWindow("Output", cv::WINDOW_NORMAL);
		  cv::imshow("Output", src_img2);
		  cv::waitKey(0);
	  }

	  std::cout << "time(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() <<std::endl;
  }

}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void WrapInputLayer(shared_ptr<Net<double> > net_,
		std::vector<cv::Mat>* input_channels)
{
	Blob<double>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	double* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_64FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Preprocess(shared_ptr<Net<double> > net_, const cv::Mat& img,
		std::vector<cv::Mat>* input_channels)
{
	Blob<double>* input_layer = net_->input_blobs()[0];
	int num_channels_ = input_layer->channels();
	cv::Size input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_64FC3);
	else
		sample_resized.convertTo(sample_float, CV_64FC1);

//	cv::Mat sample_normalized;
//	cv::subtract(sample_float, mean_, sample_normalized);
//	sample_normalized = sample_resized;

	cv::Mat sample_normalized = sample_float;

	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<double*>(input_channels->at(0).data)
			== net_->input_blobs()[0]->cpu_data())
	<< "Input channels are not wrapping the input layer of the network.";
}

template <typename Dtype>
string OutputOfBlobByName(shared_ptr<Net<Dtype> > net_, const string& blob_name)
{
	shared_ptr<Blob<Dtype> > blob = net_->blob_by_name(blob_name);
	const Dtype* begin = blob->cpu_data();
	const Dtype* end = begin + blob->channels();
	std::vector<Dtype> v(begin, end);
	std::stringstream ss;
	ss << blob_name << ": ";
	for(size_t i = 0; i < v.size(); ++i)
	{
		if(i != 0)
			ss << ",";
		ss << v[i];
	}
	return ss.str();
}

// mimic Matlab's textread()
// TODO: replace with a better synchronized method if possible
std::vector<string> TextRead(string filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return vector<string>();
	}

	std::vector<string> lines;
	string line;
	while (std::getline(ifs, line)) {
		lines.push_back(line);
	}
	return lines;
}

