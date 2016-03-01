/*
Name       : Ravi Kant
e-mail     : rkant@usc.edu	

SFM from two views

 */
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv/highgui.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

struct Frame{
	Mat frame_img;
	vector<Point3d> points_3d;
	vector<Point2d> points_2d;
	Mat descriptors;
};
vector<Frame> pointCloudsFrames;


Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
		Matx34d P,		//camera 1 matrix
		Point3d u1,		//homogenous image point in 2nd camera
		Matx34d P1		//camera 2 matrix
)
												{

	Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),
			u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),
			u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),
			u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
	);
	Matx41d B(-(u.x*P(2,3)	-P(0,3)),
			-(u.y*P(2,3)	-P(1,3)),
			-(u1.x*P1(2,3)	-P1(0,3)),
			-(u1.y*P1(2,3)	-P1(1,3)));

	Mat_<double> X;
	solve(A,B,X,DECOMP_SVD);

	return X;
												}

double TriangulatePoints(
		const vector<Point2f>& pt_set1,
		const vector<Point2f>& pt_set2,
		const Mat& K,
		const Mat& Kinv,
		const Matx34d& P,
		const Matx34d& P1,
		int frameNo)
{
	vector<double> reproj_error;
	vector<Point3d> points_3d;

	for (unsigned int i=0; i<pt_set1.size(); i++) {
		//convert to normalized homogeneous coordinates
		cout<<i<<"\n";
		Point2f kp = pt_set1[i];
		Point3d u(kp.x,kp.y,1.0);
		Mat_<double> um = Kinv * Mat_<double>(u);
		cout<<"\tgot um"<<"\n";
		u = um.at<Point3d>(0);
		Point2f kp1 = pt_set2[i];
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = Kinv * Mat_<double>(u1);
		u1 = um1.at<Point3d>(0);
		cout<<"\tgot um1"<<"\n";
		//triangulate
		Mat_<double> X = LinearLSTriangulation(u,P,u1,P1);
		if(X.at<double>(2,0) < 0){
			points_3d.clear();
			return 0;
		}
		cout<<"\tgot X"<<"\n";
		//calculate reprojection error
		cout<<K<<"\n"<<Mat(P1)<<"\n"<<X<<"\n";

		double data_x_homo[4]={X(0,0),X(1,0),X(2,0),1};
		Mat X_homo(4,1,CV_64F,data_x_homo);

		Mat_<double> xPt_img = K * Mat(P1) * X_homo;
		cout<<"\tgot reprojection"<<"\n";
		Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
		reproj_error.push_back(norm(xPt_img_-kp1));
		//store 3D point

		points_3d.push_back(Point3d(X(0,0),X(1,0),X(2,0)));
		cout<<"\tpushed to cloud"<<"\n";
	}
	//return mean reprojection error
	Scalar me = mean(reproj_error);
	cout<<"\nCloud Complete\n";
	pointCloudsFrames[frameNo].points_3d = points_3d;

}


int main()
{
	Mat right_frame_orignal = imread("robo1.jpg");
	Mat left_frame_orignal  = imread("robo2.jpg");

	// resize frames to reduce size; reduces computation
	Mat right_frame_resized, left_frame_resized;

	resize(right_frame_orignal, right_frame_resized, Size(), 0.3, 0.3, INTER_AREA);
	resize(left_frame_orignal, left_frame_resized, Size(), 0.3, 0.3, INTER_AREA);

	// convert to gray image
	Mat right_frame, left_frame;
	cvtColor(right_frame_resized, right_frame, CV_BGR2GRAY);
	cvtColor(left_frame_resized, left_frame, CV_BGR2GRAY);

	// locate points of interest using FAST
	FastFeatureDetector ffd;
	vector<KeyPoint> right_KeyPoints;
	vector<KeyPoint> left_KeyPoints;

	ffd.detect(right_frame, right_KeyPoints);
	ffd.detect(left_frame, left_KeyPoints);

	//------ Calc Optical Flow from left_frame to right_frame -------
	// For this we need to specify the points, we want the OF for

	vector<Point2f> left_KeyPoints_locations;
	for(int i = 0; i < left_KeyPoints.size(); i++) {
		left_KeyPoints_locations.push_back(left_KeyPoints[i].pt);
	}

	// we track lefttFrameKeyPoints and find their location in right_frame
	// this location is stored in OpFlow_leftKeyPoints_toRightFrame_location
	vector<Point2f> OpFlow_leftKeyPoints_toRightFrame_location;
	vector<uchar> status;
	vector<float> error;
	calcOpticalFlowPyrLK(left_frame, right_frame, left_KeyPoints_locations, OpFlow_leftKeyPoints_toRightFrame_location, status, error);

	// filter out points with high error
	vector<int> index_acceptableOptFlow;
	vector<Point2f> acceptableOptFlow_locations;
	for(int i = 0; i < OpFlow_leftKeyPoints_toRightFrame_location.size(); i++) {
		if(status[i] == 1 && error[i] < 12) {
			index_acceptableOptFlow.push_back(i);
			acceptableOptFlow_locations.push_back(OpFlow_leftKeyPoints_toRightFrame_location[i]);
		}
		else
			status[i] = 0;
	}
	// we will match the acceptable "OpFlow_leftKeyPoints_toRightFrame_location" with
	// the KeyPoints we had found in right_frame.
	// we do this by looking in the vicinity of the each "acceptableOptFlow_locations"
	// when we have a single match we keep it, when two match we keep the better of the two and if
	// there are more than 2 KeyPoints in the vicinity, we reject the point.

	BFMatcher bf_matcher(NORM_L2);
	vector<vector<DMatch> > matches;

	// get right KepPoint locations
	vector<Point2f> right_KeyPoints_locations;
	for(int i = 0; i < right_KeyPoints.size(); i++) {
		right_KeyPoints_locations.push_back(right_KeyPoints[i].pt);
	}

	// look for OF points in vicinity of each right_KeyPoint
	Mat right_KeyPoints_locations_matrix = Mat(right_KeyPoints_locations).reshape(1,right_KeyPoints_locations.size());
	Mat acceptableOptFlow_locations_matrix = Mat(acceptableOptFlow_locations).reshape(1,acceptableOptFlow_locations.size());
	bf_matcher.radiusMatch(right_KeyPoints_locations_matrix, acceptableOptFlow_locations_matrix, matches, 2.0f);

	vector<DMatch> selectedMatches;
	set<int> selectedOptFlowPoints_index;
	vector<Point2f> selected_leftPoints4OptFlow_locations;
	vector<Point2f> selected_rightPoints4OptFlow_locations;
	vector<KeyPoint> selected_left_FAST_keyPoints;
	vector<KeyPoint> selected_right_FAST_keyPoints;

	Frame f1,f2;
	f1.frame_img = left_frame;
	f2.frame_img = right_frame;
	pointCloudsFrames.push_back(f1);
	pointCloudsFrames.push_back(f2);

	for(int i = 0; i < matches.size(); i++) {
		DMatch bestMatch;
		if(matches[i].size() == 1)
		{
			bestMatch = matches[i][0];
		}
		else if(matches[i].size() == 2) {
			if(matches[i][0].distance < 0.7 * matches[i][1].distance)	// not too close
				bestMatch = matches[i][0];
			else												// too close, cant say which one is right
				continue;
		}
		else
			continue;											// too many matches; confusing point

		// add the match if it does not clash with any previous matches
		if(selectedOptFlowPoints_index.find(bestMatch.trainIdx) == selectedOptFlowPoints_index.end()) {
			selectedOptFlowPoints_index.insert(bestMatch.trainIdx);

			selected_rightPoints4OptFlow_locations.push_back(acceptableOptFlow_locations[bestMatch.trainIdx]);
			selected_right_FAST_keyPoints.push_back(right_KeyPoints[bestMatch.queryIdx]);

			pointCloudsFrames[0].points_2d.push_back(acceptableOptFlow_locations[bestMatch.trainIdx]);
			// to find the leftKeyPoint: the trainIndex gives the "acceptableOptFlow_locations" index
			// coorespoinding to this index we have "OpFlow_leftKeyPoints_toRightFrame_location"
			// agaist which we have the left_KeyPoints_locations
			selected_leftPoints4OptFlow_locations.push_back(left_KeyPoints_locations[index_acceptableOptFlow[bestMatch.trainIdx]]);
			pointCloudsFrames[1].points_2d.push_back(left_KeyPoints_locations[index_acceptableOptFlow[bestMatch.trainIdx]]);
			// bestMatch query should refer to left_KeyPoints
			bestMatch.queryIdx = index_acceptableOptFlow[bestMatch.trainIdx];
			selected_left_FAST_keyPoints.push_back(left_KeyPoints[index_acceptableOptFlow[bestMatch.trainIdx]]);
			selectedMatches.push_back(bestMatch);
		}
	}

	Mat image_matches;
	// for drawMatches we need the PF points in right Image to be wrapped in
	// KeyPoint format. SO we wrap the selected locations into KeyPoints
	vector<KeyPoint> disguisedOpFlowAsKeyPoint;
	for(int i = 0; i < acceptableOptFlow_locations.size(); i++) {
		KeyPoint kp;
		kp.pt = acceptableOptFlow_locations[i];
		disguisedOpFlowAsKeyPoint.push_back(kp);
	}

	namedWindow("Good Matches", WINDOW_NORMAL);
	drawMatches(left_frame, left_KeyPoints, right_frame, disguisedOpFlowAsKeyPoint, selectedMatches, image_matches, Scalar::all(-1),Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	imshow( "Good Matches", image_matches );
	waitKey(0);

	// add descriptors for selected points
	SiftDescriptorExtractor sift;
	Mat descriptors;
	sift.compute(left_frame, selected_left_FAST_keyPoints,descriptors);
	pointCloudsFrames[0].descriptors = descriptors;
	sift.compute(right_frame, selected_right_FAST_keyPoints,descriptors);
	pointCloudsFrames[1].descriptors = descriptors;
	//---------------------------- end of matching -----------------------

	// Step 2: reconstruction

	double K_data[9] = {1.0495656184032528e+03, 0.0, 6.3950000000000000e+02,
			0.0, 1.0495656184032528e+03, 3.5950000000000000e+02,
			0.0, 0.0, 1.0};

	Mat K(3,3,CV_64F,K_data);
	Mat_<double> Kinv = K.inv();
	double W_data[9] = {0.0, -1.0, 0.0,
			1.0, 0.0, 0.0,
			0.0, 0.0, 1.0};
	Mat W(3,3,CV_64F,W_data);

	// funadmental matrix
	// TODO: See if heartley transformation improves result
	Mat F = findFundamentalMat(Mat(selected_leftPoints4OptFlow_locations), Mat(selected_rightPoints4OptFlow_locations), FM_RANSAC, 0.1, 0.99);

	// Essential matrix
	Mat_<double> E = K.t() * F * K;

	SVD svd(E);

	Mat_<double> R1 = svd.u * W * svd.vt;
	Mat_<double> T1 = svd.u.col(2);

	double Wt_data[9] = {0,1,0,
			-1,0,0,
			0,0,1};
	Mat Wt(3,3,CV_64F,Wt_data);

	Mat_<double> R2 = svd.u * Wt * svd.vt;
	Mat_<double> T2 = -1 * svd.u.col(2);

	// check R1 and R2 are valid
	if(fabsf(determinant(R1))-1.0 > 1e-07) {
		cerr << "det(R1) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}
	if(fabsf(determinant(R2))-1.0 > 1e-07) {
		cerr << "det(R2) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}

	cout<<"\n\n DONE TILL HERE";
	Matx34d P0( 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0);
	Matx34d P1;
	double reProjError_0, reProjError_1, reProjError_2, reProjError_3;
	// case0: P1 = [R1 T1]
	Matx34d P1_0(R1(0,0), R1(0,1), R1(0,2), T1(0,0),
			R1(1,0), R1(1,1), R1(1,2), T1(1,0),
			R1(2,0), R1(2,1), R1(2,2), T1(2,0));

	reProjError_0 = TriangulatePoints(selected_leftPoints4OptFlow_locations,
			selected_rightPoints4OptFlow_locations,
			K,
			Kinv,
			P0,
			P1_0,
			0);
	P1 = P1_0;

	// case1: P1 = [R1 T2]
	if(reProjError_0 == 0) {
		Matx34d P1_1(R1(0,0), R1(0,1), R1(0,2), T2(0,0),
				R1(1,0), R1(1,1), R1(1,2), T2(1,0),
				R1(2,0), R1(2,1), R1(2,2), T2(2,0));

		reProjError_1 = TriangulatePoints(selected_leftPoints4OptFlow_locations,
				selected_rightPoints4OptFlow_locations,
				K,
				Kinv,
				P0,
				P1_1,
				0);
		P1 = P1_1;
	}

	// case3: P2 = [R2 T1]
	if(reProjError_1 == 0) {
		Matx34d P1_2(R2(0,0), R2(0,1), R2(0,2), T1(0,0),
				R2(1,0), R2(1,1), R2(1,2), T1(1,0),
				R2(2,0), R2(2,1), R2(2,2), T1(2,0));

		reProjError_2 = TriangulatePoints(selected_leftPoints4OptFlow_locations,
				selected_rightPoints4OptFlow_locations,
				K,
				Kinv,
				P0,
				P1_2,
				0);
		P1 = P1_2;
	}
	// case3: P1 = [R2 T2]
	if(reProjError_2 == 0) {
		Matx34d P1_3(R2(0,0), R2(0,1), R2(0,2), T2(0,0),
				R2(1,0), R2(1,1), R2(1,2), T2(1,0),
				R2(2,0), R2(2,1), R2(2,2), T2(2,0));

		reProjError_3 = TriangulatePoints(selected_leftPoints4OptFlow_locations,
				selected_rightPoints4OptFlow_locations,
				K,
				Kinv,
				P0,
				P1_3,
				0);
		P1 = P1_3;
	}

	// copy 3d points to frame 1
	pointCloudsFrames[1].points_3d = pointCloudsFrames[0].points_3d;


	// Step 3: Multi View
	cout<<"\n==============Starting MultiView===========\n";
	VideoCapture cap("robo.webm");
	if(!cap.isOpened()) {
		cout<<"\nVideo file does not exist";
		return -1;
	}

	int numFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat curFrame_orignal, curFrame_resized,curFrame, curFrame_descriptors;
	Mat prevFrame;
	double dCoeff[5] = {0.058055443802597889, -0.75762481381245350, 0.0, 0.0,1.2767832543908491};
	vector<double> distCoeffs(&dCoeff[0], &dCoeff[0]+5);
	for(int curFrameNo = 0; curFrameNo < numFrames; curFrameNo++) {
		vector<KeyPoint> curFrameKeyPoints;
		cap>>curFrame_orignal;
		resize(curFrame_orignal, curFrame_resized,Size(), 0.3, 0.3, INTER_AREA);
		cvtColor(curFrame_resized, curFrame, CV_BGR2GRAY);
		imshow("curFrame",curFrame);
		waitKey(1);
		pointCloudsFrames[curFrameNo].frame_img = curFrame;
		// extract keyPoints in current frame
		ffd.detect(curFrame, curFrameKeyPoints);
		sift.compute(curFrame,curFrameKeyPoints,curFrame_descriptors);


		vector<int>curFrameDescriptorIndex(curFrameKeyPoints.size(),-1);

		// we will match the current frame with previous 7 frames
		// to get matching points
		int startFrameNo,endFrameNo;
		if(curFrameNo - 7 < 0) {
			startFrameNo = 0;
			endFrameNo = pointCloudsFrames.size();
		}
		else {
			startFrameNo = curFrameNo - 7;
			endFrameNo = curFrameNo - 1;
		}
		cout<<"start: "<<startFrameNo<<"end "<<endFrameNo<<"\n";
		// the 1st two frames are taken up by baseline cases
		// so we start search from 2nd frame
		vector<vector<DMatch> > frameMatches;
		vector<DMatch> frameBestMatches;
		Mat frameDescriptors;
		Frame newFrame;
		newFrame.frame_img = curFrame;
		for(int searchFrameNo = startFrameNo; searchFrameNo < endFrameNo; searchFrameNo++) {
			//select keyPoints in curFrame that have not been matched yet

			Mat descriptorsToMatch;
			vector<KeyPoint> keyPointsToMatch;
			vector<int> descriptorToMatchIndex;
			for(int j = 0; j < curFrameKeyPoints.size(); j++) {
				if(curFrameDescriptorIndex[j] == -1){
					descriptorsToMatch.push_back(curFrame_descriptors.row(j));
					keyPointsToMatch.push_back(curFrameKeyPoints[j]);
					descriptorToMatchIndex.push_back(j);
				}
			}

			bf_matcher.knnMatch(descriptorsToMatch, pointCloudsFrames[searchFrameNo].descriptors, frameMatches, 1);
			cout<<"\n===========>frameMatches"<<frameMatches.size();
			for(int j = 0; j < frameMatches.size(); j++) {
				cout<<"\n\t"<<frameMatches[j][0].distance;
				if(frameMatches[j][0].distance < 150) {
					frameBestMatches.push_back(frameMatches[j][0]);
					curFrameDescriptorIndex[descriptorToMatchIndex[j]] = 1;
					newFrame.points_2d.push_back(keyPointsToMatch[frameMatches[j][0].queryIdx].pt);
					newFrame.descriptors.push_back(descriptorsToMatch.row(frameMatches[j][0].queryIdx));
					newFrame.points_3d.push_back(pointCloudsFrames[searchFrameNo].points_3d[frameMatches[j][0].trainIdx]);
				}
			}
			vector<KeyPoint> fakeKeyPoints;
			for(int j = 0; j < pointCloudsFrames[searchFrameNo].points_2d.size(); j++) {
				KeyPoint temp;
				temp.pt = pointCloudsFrames[searchFrameNo].points_2d[j];
				fakeKeyPoints.push_back(temp);
			}
			Mat imgMatches;
			cout<<"\nSizes "<<curFrameKeyPoints.size()<<" "<<fakeKeyPoints.size()<<" "<<frameBestMatches.size()<<"\n";
			namedWindow("Good Frame Matches", WINDOW_NORMAL);
			drawMatches(curFrame, keyPointsToMatch, pointCloudsFrames[curFrameNo].frame_img, fakeKeyPoints,frameBestMatches, imgMatches, Scalar::all(-1),Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
			imshow( "Good Frame Matches", imgMatches );
			waitKey(0);
		}
		cout<<"\n\ncame out\n\n";
		cout<<"\n"<<newFrame.points_3d.size()<<" "<<newFrame.points_2d.size()<<"\n";
		// selectPoints for nPn
		Mat_<double> rvec, R_frame, t_frame;
		solvePnPRansac(newFrame.points_3d, newFrame.points_2d, K, distCoeffs, R_frame, t_frame, false);

		Rodrigues(rvec, R_frame);
		Matx34d P1_frame = Matx34d(R_frame.at<double>(0,0),R_frame.at<double>(0,1),R_frame.at<double>(0,2),t_frame.at<double>(0,0),
				R_frame.at<double>(1,0),R_frame.at<double>(1,1),R_frame.at<double>(1,2),t_frame.at<double>(1,0),
				R_frame.at<double>(2,0),R_frame.at<double>(2,1),R_frame.at<double>(2,2),t_frame.at<double>(2,0));
cout<<"\n\t1\n";
		// triangulate those points for which no matches were found in previous frames
		vector<KeyPoint> remainingKeyPoints;
		for(int j = 0; j < curFrameKeyPoints.size(); j++) {
			if(curFrameDescriptorIndex[j] == -1){
				remainingKeyPoints.push_back(curFrameKeyPoints[j]);
			}
		}
		if(curFrameNo == 0){
			prevFrame = right_frame;
		}
		vector<KeyPoint> prevKeyPoints;
		for(int j = 0; j < pointCloudsFrames[curFrameNo+1].points_2d.size(); j++ ) {
			KeyPoint temp;
			temp.pt = pointCloudsFrames[curFrameNo+1].points_2d[j];
			prevKeyPoints.push_back(temp);
		}

cout<<"\n\nReached here\n";
		index_acceptableOptFlow.clear();
		acceptableOptFlow_locations.clear();
		OpFlow_leftKeyPoints_toRightFrame_location.clear();
		status.clear();
		error.clear();

		calcOpticalFlowPyrLK(prevFrame, curFrame, prevKeyPoints, OpFlow_leftKeyPoints_toRightFrame_location, status, error);

		for(int i = 0; i < OpFlow_leftKeyPoints_toRightFrame_location.size(); i++) {
			if(status[i] == 1 && error[i] < 12) {
				index_acceptableOptFlow.push_back(i);
				acceptableOptFlow_locations.push_back(OpFlow_leftKeyPoints_toRightFrame_location[i]);
			}
			else
				status[i] = 0;
		}

		// look for OF points in vicinity of each right_KeyPoint
		Mat right_KeyPoints_locations_matrix = Mat(remainingKeyPoints).reshape(1,remainingKeyPoints.size());
		Mat acceptableOptFlow_locations_matrix = Mat(acceptableOptFlow_locations).reshape(1,acceptableOptFlow_locations.size());
		bf_matcher.radiusMatch(right_KeyPoints_locations_matrix, acceptableOptFlow_locations_matrix, matches, 2.0f);

		selectedMatches.clear();
		selectedOptFlowPoints_index.clear();
		selected_leftPoints4OptFlow_locations.clear();
		selected_rightPoints4OptFlow_locations.clear();
		selected_left_FAST_keyPoints.clear();
		selected_right_FAST_keyPoints.clear();

		for(int i = 0; i < matches.size(); i++) {
			DMatch bestMatch;
			if(matches[i].size() == 1)
			{
				bestMatch = matches[i][0];
			}
			else if(matches[i].size() == 2) {
				if(matches[i][0].distance < 0.7 * matches[i][1].distance)	// not too close
					bestMatch = matches[i][0];
				else												// too close, cant say which one is right
					continue;
			}
			else
				continue;											// too many matches; confusing point

			// add the match if it does not clash with any previous matches
			if(selectedOptFlowPoints_index.find(bestMatch.trainIdx) == selectedOptFlowPoints_index.end()) {
				selectedOptFlowPoints_index.insert(bestMatch.trainIdx);

				selected_rightPoints4OptFlow_locations.push_back(acceptableOptFlow_locations[bestMatch.trainIdx]);
				selected_right_FAST_keyPoints.push_back(remainingKeyPoints[bestMatch.queryIdx]);
				newFrame.points_2d.push_back(remainingKeyPoints[bestMatch.queryIdx].pt);

				// to find the leftKeyPoint: the trainIndex gives the "acceptableOptFlow_locations" index
				// coorespoinding to this index we have "OpFlow_leftKeyPoints_toRightFrame_location"
				// agaist which we have the left_KeyPoints_locations
				selected_leftPoints4OptFlow_locations.push_back(left_KeyPoints_locations[index_acceptableOptFlow[bestMatch.trainIdx]]);

				// bestMatch query should refer to left_KeyPoints
				bestMatch.queryIdx = index_acceptableOptFlow[bestMatch.trainIdx];
				selected_left_FAST_keyPoints.push_back(left_KeyPoints[index_acceptableOptFlow[bestMatch.trainIdx]]);
				selectedMatches.push_back(bestMatch);
			}
		}
		pointCloudsFrames.push_back(newFrame);
		reProjError_0 = TriangulatePoints(selected_leftPoints4OptFlow_locations,
				selected_rightPoints4OptFlow_locations,
				K,
				Kinv,
				P0,
				P1_frame,
				curFrameNo+2);





		prevFrame = curFrame;
	}
}// end of main
