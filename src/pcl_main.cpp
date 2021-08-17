#include <iostream>
#include <fstream> 

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp> 
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include<pcl/visualization/cloud_viewer.h>



using namespace std; 

string left_image = "../images/left.png";
string right_image = "../images/right.png";

int main(int argc, char **argv){
	
    vector<Eigen::Isometry3d> poses;

	// Camera Parameters
	double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
	double b = 0.573;
	
	cv::Mat left  = cv::imread(left_image, 0);
	cv::Mat right = cv::imread(right_image, 0);
	
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
	cv::Mat disparity_sgbm, disparity;
	sgbm->compute(left, right, disparity_sgbm);
	disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    Eigen::Quaterniond q(1, 0, 0, 0);
    Eigen::Isometry3d T(q);
    T.pretranslate(Eigen::Vector3d(0, 0, 0));
    poses.push_back(T);

    typedef pcl::PointXYZ PointT; 
    typedef pcl::PointCloud<PointT> PointCloud; 

    PointCloud::Ptr pointCloud(new PointCloud);
    PointCloud::Ptr current(new PointCloud); 

    cv::Mat disp = disparity; 

    for (int v = 0; v < left.rows; v++){
        for (int u = 0; u < left.cols; u++){
            if (disparity.at<float>(v, u) <=0.0 || disparity.at<float>(v, u) >=96.0){
				continue;
			}
            Eigen::Vector3d point; 
            point[2] = fx * b / (disparity.at<float>(v,u));
            point[0] = (u - cx) * point[2] / fx;
            point[1] = (v - cy) * point[2] / fy;
            Eigen::Vector3d pointWorld = poses[0] * point; 

            PointT p;
            p.x = pointWorld[0];
            p.y = pointWorld[1];
            p.z = pointWorld[2];
            current->points.push_back(p);
        }
    }
    PointCloud::Ptr tmp(new PointCloud);
    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    statistical_filter.setMeanK(90);
    statistical_filter.setStddevMulThresh(1.0);
    statistical_filter.setInputCloud(current);
    statistical_filter.filter(*tmp);
    (*pointCloud) += *tmp;

    pointCloud->is_dense = false;


    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.1;
    voxel_filter.setLeafSize(resolution, resolution, resolution);       // resolution
    PointCloud::Ptr tmpq(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmpq);
    tmpq->swap(*pointCloud);

    // Saving PCD
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);

    // Visualize PCD
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>); 

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("map.pcd", *cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file \n"); 
        return -1;
    }
    pcl::visualization::CloudViewer viewer ("Simple Cloud"); 
    viewer.showCloud (cloud);
    while (!viewer.wasStopped()){
        
    }



}