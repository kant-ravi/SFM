/*
Name       : Ravi Kant
USC ID     : 7945-0425-48
e-mail     : rkant@usc.edu
Submission : Jan 11, 2016

Input Format: programName Data_Location ClassA_name number_of_classA_samples
				classB_name number_of_classB_samples number_of_test_samples

 */

#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
//#include "/home/ravi/opencv-2.4.11/modules/viz/include/opencv2/viz/types.hpp"
#include <opencv/highgui.h>
#include <iostream>
#include <string>
#include <fstream>

#include <usr/include/pcl-1.7/pcl/io/pcdio.h>
#include "usr/include/pcl-1.7/pcl/pointtypes.h""
using namespace cv;
using namespace std;


int main()
{
	Mat image1 = imread("/home/ravi/workspace/MyProjects/3DReconstruction/1.jpg",0);
	Mat image2 = imread("/home/ravi/workspace/MyProjects/3DReconstruction/2.jpg",0);

	Mat im1,im2;
	resize(image1,image1,Size(),0.3,0.3,INTER_LINEAR);
	resize(image2,image2,Size(),0.3,0.3,INTER_LINEAR);
	//	imshow("img",image1);
	//	waitKey(0);
	Mat flow;
	double pyr_scale = 0.5;
	int levels = 3;
	int winsize = 15;
	int iterations = 3;
	int poly_n = 5;
	double poly_sigma = 1.2;
	int flags = 0;
	calcOpticalFlowFarneback(image1, image2, flow, pyr_scale, levels, winsize, iterations,  poly_n,  poly_sigma, flags);
	cvtColor(image1, image1, COLOR_GRAY2BGR);

	vector<Point2f> image1_pts,image2_pts;
	for(int y = 0; y < image1.rows; y += 1)
		for(int x = 0; x < image1.cols; x += 1)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(image1, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
					Scalar(0, 255, 0));
			Point diff = Point(x,y) - Point(cvRound(x+fxy.x), cvRound(y+fxy.y));
			float dist = sqrt(diff.x*diff.x + diff.y*diff.y);
			if(dist!=0)
			{
				image1_pts.push_back(Point(x,y));
				image2_pts.push_back(Point(cvRound(x+fxy.x), cvRound(y+fxy.y)));
			}
			circle(image1, Point(x,y), 2, Scalar(0, 255, 0), -1);
			//if(x<3 && y<3)
			//cout<<image1_pts;
		}
	//	imshow("flow", image1);
	//waitKey(0);
	//cout<<image1_pts;
	Mat F =  findFundamentalMat  (Mat(image1_pts), Mat(image2_pts), FM_RANSAC, 0.1, 0.99);
	cout<<"done\n";
	cout<<F;
	cout<<F.type()<<"==========\n";
	double data[9] = {1.0495656184032528e+03, 0.0, 6.3950000000000000e+02,
			0.0, 1.0495656184032528e+03, 3.5950000000000000e+02,
			0.0, 0.0, 1.0};
	/*
	 * {1042.03866, 0.0, 639.64111,
			0.0, 1045.03966, 383.97325,
			0.0, 0.0, 1.0};
			{1.0495656184032528e+03, 0.0, 6.3950000000000000e+02,
			0.0, 1.0495656184032528e+03, 3.5950000000000000e+02,
			0.0, 0.0, 1.0};
	 */
	Mat K(3,3,CV_64F,data);
	cout<<"\n\nK"<<K;
	Mat Ktrans;
	transpose(K,Ktrans);
	cout<<"=====2=====\n";
	Mat essentialMatrix = F * K;
	cout<<"=====3=====\n";
	essentialMatrix = Ktrans * essentialMatrix;
	cout<<"=====4=====\n";
	cout<<"done\n";
	cout<<essentialMatrix;
	cout<<"\n=====5=====\n";
	SVD(essentialMatrix, 0 );
	Mat w,u,vt;
	SVD::compute(essentialMatrix,  w,  u,  vt, 0 );

	double data_w[9] = {0.0,-1.0,0.0,
			1.0,0.0,0.0,
			0.0,0.0,1.0};
	Mat w_new(3,3,CV_64F,data_w);
	Mat R = u * w_new * vt;
	Mat t = u.col(2);
	Mat P1(3,4,CV_64F,0), P2;
	cout<<"\n\nR\n"<<R<<"\n\n";
	if(fabsf(determinant(R))-1 > 1e-07){
		cout<<"Invalid R\n";
	}
	else{
		hconcat(R, t, P1);
	}
	cout<<"\n DONE"<<P1;
	double data3[12]={1.0,0.0,0.0,0.0,
			0.0,1.0,0.0,0.0,
			0.0,0.0,1.0,0.0};
	Mat P0(3,4,CV_64F,data3);

	Mat KP0 = K * P0;
	Mat KP1 = K * P1;
	Mat pointCloud;
	for(int i = 0; i < image1_pts.size(); i ++){
		Point3f u_c1(image1_pts[i].x,image1_pts[i].y,1);
		Point3f u_c2(image2_pts[i].x,image2_pts[1].y,1);

		double data_A[12] = {u_c1.x*KP0.at<double>(2,0)-KP0.at<double>(0,0),u_c1.x*KP0.at<double>(2,1)-KP0.at<double>(0,1),u_c1.x*KP0.at<double>(2,2)-KP0.at<double>(0,2),
				u_c1.y*KP0.at<double>(2,0)-KP0.at<double>(1,0),u_c1.y*KP0.at<double>(2,1)-KP0.at<double>(1,1),u_c1.y*KP0.at<double>(2,2)-KP0.at<double>(1,2),
				u_c2.x*KP1.at<double>(2,0)-KP1.at<double>(0,0), u_c2.x*KP1.at<double>(2,1)-KP1.at<double>(0,1),u_c2.x*KP1.at<double>(2,2)-KP1.at<double>(0,2),
				u_c2.y*KP1.at<double>(2,0)-KP1.at<double>(1,0), u_c2.y*KP1.at<double>(2,1)-KP1.at<double>(1,1),u_c2.y*KP1.at<double>(2,2)-KP1.at<double>(1,2)
		};

		double data_B[4] ={-(u_c1.x*KP0.at<double>(2,3)-KP0.at<double>(0,3)),
				-(u_c1.y*KP0.at<double>(2,3)-KP0.at<double>(1,3)),
				-(u_c2.x*KP1.at<double>(2,3)-KP1.at<double>(0,3)),
				-(u_c2.y*KP1.at<double>(2,3)-KP1.at<double>(1,3))};

		Mat A(4,3,CV_64F,data_A);
		Mat B(4,1,CV_64F,data_B);

		Mat_<double> X;
		solve(A,B,X,DECOMP_SVD);

		pointCloud.push_back(Point3d(X.at<double>(0,0),X.at<double>(0,1),X.at<double>(0,2)));




	}

	ofstream fout;
	fout.open("3dCloud.txt");
	//	 for(int i =0; i < pointCloud.rows;i++)
	//	 fout<<pointCloud.row(i)<<"\n";
	fout<<pointCloud;
	fout.close();



	cout<<"**********************************";

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	void PopulatePCLPointCloud(const vector<Point3d>& pointcloud,
			const std::vector<cv::Vec3b>& pointcloud_RGB
	)
	//Populate point cloud
	{
		cout<<"Creating point cloud...";
		cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (unsigned int i=0; i<pointcloud.size(); i++) {
			// get the RGB color value for the point
			Vec3b rgbv(255,255,255);
			if (pointcloud_RGB.size() >= i) {
				rgbv = pointcloud_RGB[i];
			}
			// check for erroneous coordinates (NaN, Inf, etc.)
			if (pointcloud[i].x != pointcloud[i].x || isnan(pointcloud[i].x) ||
					pointcloud[i].y != pointcloud[i].y || isnan(pointcloud[i].y) ||
					pointcloud[i].z != pointcloud[i].z || isnan(pointcloud[i].z) ||
					fabsf(pointcloud[i].x) > 10.0 ||
					fabsf(pointcloud[i].y) > 10.0 ||
					fabsf(pointcloud[i].z) > 10.0) {
				continue;
			}
			pcl::PointXYZRGB pclp;
			// 3D coordinates
			pclp.x = pointcloud[i].x;
			pclp.y = pointcloud[i].y;
			pclp.z = pointcloud[i].z;
			// RGB color, needs to be represented as an integer
			uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 |
			(uint32_t)rgbv[0]);
			pclp.rgb = *reinterpret_cast<float*>(&rgb);
			cloud->push_back(pclp);
			}
			cloud->width = (uint32_t) cloud->points.size(); // number of points
			cloud->height = 1; // a list of points, one row of data
			}
		}// end of main
