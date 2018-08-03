#pragma once

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utils/filesystem.hpp"

#include <stdio.h>
#include <string.h>

#include <time.h>

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


class Calibration
{
public:
	//Calibration() {};
	//~Calibration() {};

public:
	enum { DETECTION, CAPTURING, CALIBRATED };

	enum Pattern {CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID};

	explicit Calibration(
		const std::string& imgsDirector,
		const std::string& outputFilename,
		Size boardSize, 
		double squareSize
	);

	static double computeReprojectionErrors(
		const vector<vector<Point3f> >& objectPoints,
		const vector<vector<Point2f> >& imagePoints,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const Mat& cameraMatrix, const Mat& distCoeffs,
		vector<float>& perViewErrors);

	static void calcChessboardCorners(Size boardSize, float squareSize, 
		vector<Point3f>& corners, Pattern patternType = CHESSBOARD);

	// calibration functions
	bool doCalibration();

	cv::Mat getCameraMatrix() const;
	cv::Mat getDistCoeffsMatrix() const;
	cv::Mat getExtrinsicsBigMat() const;
	vector<int> getFoundCheeseBoardVec() const;

protected:
	void saveCameraParams(const string& filename,
		Size imageSize, Size boardSize,
		float squareSize, float aspectRatio, int flags,
		const Mat& cameraMatrix, const Mat& distCoeffs,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const vector<float>& reprojErrs,
		const vector<vector<Point2f> >& imagePoints,
		double totalAvgErr);

	bool runCalibration(vector<vector<Point2f> > imagePoints,
		Size imageSize, Size boardSize, Pattern patternType,
		float squareSize, float aspectRatio,
		int flags, Mat& cameraMatrix, Mat& distCoeffs,
		vector<Mat>& rvecs, vector<Mat>& tvecs,
		vector<float>& reprojErrs,
		double& totalAvgErr);

	bool runAndSave(const string& outputFilename,
		const vector<vector<Point2f> >& imagePoints,
		Size imageSize, Size boardSize, Pattern patternType, float squareSize,
		float aspectRatio, int flags, Mat& cameraMatrix,
		Mat& distCoeffs, bool writeExtrinsics, bool writePoints);

private:
	Size boardSize; 
	float squareSize = 0.010;   //   0.01 m , 10 ms
	float aspectRatio = 1.0;

	bool undistortImage = false;
	int flags = 0;
	bool flipVertical = false;
	bool showUndistorted = false;

	clock_t prevTimestamp = 0;

	string imgsDirectory;
	string outputFilename;

	bool writeExtrinsics = true;
	bool writePoints = true;

	Pattern pattern = CHESSBOARD;
	int mode = CAPTURING;

	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;

	// user
	vector<int> foundCheeseBoardVec;
	Mat extrinsicsBigMat;
};

