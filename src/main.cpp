#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unisrd.h> 

using namespace std; 
using namespace Eigen; 

string left_image = "../images/left.png";
string right_image = "../images/right.png";

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud){

	if (pointcloud.empty()){
		cerr << "Bhai kuch nhi hai! :(";
		return;
	}	

	pangolin::CreateWindowAndBind("My 3D Reconstruction", 1024, 768); 




}




int main(int argc, char **argv){
	
	// Camera Parameters
	double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
	double b = 0.573;
	
	cv::Mat left  = cv::imread(left_image, 0);
	cv::Mat right = cv::imread(right_image, 0);
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
	cv::Mat disparity_sgbm, disparity;
	sgbm->compute(left, right, disparity_sgbm);
	disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

	vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud; 

	for (int v = 0; v < left.rows; v++){

		for (int u = 0; u<left.cols; u++){
			if (disparity.at<float>(u,v) <=0.0 || disparity.at<float>(u,v) >=96.0){
				continue;
			}

			Vector4d point(0, 0, 0, left.at<uchar>(u,v) / 255.0);
			
			double x = (u - cx) / fx;
			double y = (v - cy) / fy;
			double depth = fx * b / (disparity.at<float>(v,u));
			point[0] = x * depth;
			point[1] = y * depth;
			point[2] = depth;

			pointcloud.push_back(point);
		
		}
	}

	cv::imshow("disparity", disparity / 96.0);
	cv::waitKey(0);
	showPointCloud(pointCloud);
	return 0;
}


