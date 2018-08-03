#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>


#include <iostream>
#include <fstream>
#include "opencv2/core/utils/filesystem.hpp"

using namespace cv;
using namespace std;

const char * usage =
" \nexample command line for calibration from a live feed.\n"
"   calibration  -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe\n"
" \n"
" example command line for calibration from a list of stored images:\n"
"   imagelist_creator image_list.xml *.png\n"
"   calibration -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe image_list.xml\n"
" where image_list.xml is the standard OpenCV XML/YAML\n"
" use imagelist_creator to create the xml or yaml list\n"
" file consisting of the list of strings, e.g.:\n"
" \n"
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<images>\n"
"view000.png\n"
"view001.png\n"
"<!-- view002.png -->\n"
"view003.png\n"
"view010.png\n"
"one_extra_view.jpg\n"
"</images>\n"
"</opencv_storage>\n";

const char* liveCaptureHelp =
"When the live video from camera is used as input, the following hot-keys may be used:\n"
"  <ESC>, 'q' - quit the program\n"
"  'g' - start capturing images\n"
"  'u' - switch undistortion on/off\n";

static void help()
{
	printf("This is a camera calibration sample.\n"
		"Usage: calibration\n"
		"     -w=<board_width>         # the number of inner corners per one of board dimension\n"
		"     -h=<board_height>        # the number of inner corners per another board dimension\n"
		"     [-pt=<pattern>]          # the type of pattern: chessboard or circles' grid\n"
		"     [-n=<number_of_frames>]  # the number of frames to use for calibration\n"
		"                              # (if not specified, it will be set to the number\n"
		"                              #  of board views actually available)\n"
		"     [-d=<delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
		"                              # (used only for video capturing)\n"
		"     [-s=<squareSize>]       # square size in some user-defined units (1 by default)\n"
		"     [-o=<out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
		"     [-op]                    # write detected feature points\n"
		"     [-oe]                    # write extrinsic parameters\n"
		"     [-zt]                    # assume zero tangential distortion\n"
		"     [-a=<aspectRatio>]       # fix aspect ratio (fx/fy)\n"
		"     [-p]                     # fix the principal point at the center\n"
		"     [-v]                     # flip the captured images around the horizontal axis\n"
		"     [-V]                     # use a video file, and not an image list, uses\n"
		"                              # [input_data] string for the video file name\n"
		"     [-su]                    # show undistorted images after calibration\n"
		"     [input_data]             # input data, one of the following:\n"
		"                              #  - text file with a list of the images of the board\n"
		"                              #    the text file can be generated with imagelist_creator\n"
		"                              #  - name of video file with a video of the board\n"
		"                              # if input_data not specified, a live view from the camera is used\n"
		"\n");
	printf("\n%s", usage);
	printf("\n%s", liveCaptureHelp);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


// 枚举变量, 静态函数

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

static double computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
			cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize,
	float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
	corners.resize(0);

	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float(j*squareSize),
					float(i*squareSize), 0));
		break;

	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float((2 * j + i % 2)*squareSize),
					float(i*squareSize), 0));
		break;

	default:
		CV_Error(Error::StsBadArg, "Unknown pattern type\n");
	}
}

static bool runCalibration(vector<vector<Point2f> > imagePoints,
	Size imageSize, Size boardSize, Pattern patternType,
	float squareSize, float aspectRatio,
	int flags, Mat& cameraMatrix, Mat& distCoeffs,
	vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs,
	double& totalAvgErr)
{
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (flags & CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = aspectRatio;

	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	//calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
	auto& corners = objectPoints[0];
	corners.resize(0);
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(float(j*squareSize),
				float(i*squareSize), 0));


	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, flags | CALIB_FIX_K4 | CALIB_FIX_K5);
	///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);

	printf("RMS error reported by calibrateCamera: %g\n", rms);

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}


static void saveCameraParams(const string& filename,
	Size imageSize, Size boardSize,
	float squareSize, float aspectRatio, int flags,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const vector<float>& reprojErrs,
	const vector<vector<Point2f> >& imagePoints,
	double totalAvgErr)
{
	FileStorage fs(filename, FileStorage::WRITE);

	time_t tt;
	time(&tt);
	struct tm *t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;

	if (flags & CALIB_FIX_ASPECT_RATIO)
		fs << "aspectRatio" << aspectRatio;

	if (flags != 0)
	{
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
		//cvWriteComment( *fs, buf, 0 );
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	//// 增加旋转、平移
	//fs << "rvecs" << rvecs;
	//fs << "distortion_coefficients" << distCoeffs;

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); i++)
		{
			Mat r = bigmat(Range(i, i + 1), Range(0, 3));
			Mat t = bigmat(Range(i, i + 1), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();	// 旋转向量 -> Rodrigues变换 -> 旋转矩阵  
			t = tvecs[i].t();	// 旋转矩阵

								//Mat r_matrix;
								//cv::Rodrigues(r, r_matrix);
		}
		//cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
		fs << "extrinsic_parameters" << bigmat;
	}

	if (!imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}
}

static bool runAndSave(const string& outputFilename,
	const vector<vector<Point2f> >& imagePoints,
	Size imageSize, Size boardSize, Pattern patternType, float squareSize,
	float aspectRatio, int flags, Mat& cameraMatrix,
	Mat& distCoeffs, bool writeExtrinsics, bool writePoints)
{
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
		aspectRatio, flags, cameraMatrix, distCoeffs,
		rvecs, tvecs, reprojErrs, totalAvgErr);
	printf("%s. avg reprojection error = %.2f\n",
		ok ? "Calibration succeeded" : "Calibration failed",
		totalAvgErr);

	if (ok)
		saveCameraParams(outputFilename, imageSize,
			boardSize, squareSize, aspectRatio,
			flags, cameraMatrix, distCoeffs,
			writeExtrinsics ? rvecs : vector<Mat>(),
			writeExtrinsics ? tvecs : vector<Mat>(),
			writeExtrinsics ? reprojErrs : vector<float>(),
			writePoints ? imagePoints : vector<vector<Point2f> >(),
			totalAvgErr);
	return ok;
}


//////////////////////////////////////////////////////////////////////////


int calibration()
{
	// 读取图片
	vector<String> imageList;

	std::string path = R"(C:\Users\wanggao\Desktop\eyeinhand\图片\)"; ;
	//std::string path = R"(E:\Microsoft Visual Studio 2015\OpenCv340Test\x64\Debug\cal_imgs\)";
	cv::glob(path + "/各角度/*.*", imageList);
	if (imageList.size() == 0) {
		std::cout << "no images." << std::endl;	return -1;
	}


	std::ofstream sucessImgs("out_success_file.txt");

	//path = R"(C:\Users\wanggao\Desktop\eyeinhand\图片\各角度增强\)";
	//cv::utils::fs::createDirectories(path);
	//for (auto& il : imageList) {
	//	int nPosS = il.find_last_of('\\');
	//	int nPosE = il.find_last_of('.');
	//	string f = il.substr(nPosS + 1, nPosE - nPosS - 1);
	//	Mat img, tmp;
	//	img = imread(il);
	//	//resize(img, tmp, img.size() / 2);
	//	tmp = img * 1.5 + 10;
	//	imwrite(path + f + ".jpg", tmp);
	//}

	int nframes = (int)imageList.size();

	// 参数定义
	Size boardSize(11, 8);     // 	
	float squareSize = 10 /*.010*/;   //   0.01 m , 10 ms
	float aspectRatio = 1.0;

	bool undistortImage = false;
	int flags = 0;
	bool flipVertical = false;
	bool showUndistorted = false;

	clock_t prevTimestamp = 0;

	string outputFilename("out_camera_data.yml");
	bool writeExtrinsics = true;
	bool writePoints = false;

	Pattern pattern = CHESSBOARD;
	int mode = CAPTURING;

	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;

	if (showUndistorted) {
		namedWindow("Image View", 1);
		resizeWindow("Image View", { 800,800 });
	}

	// 标定
	std::cout << "Process: " << std::endl;
	for (int i = 0;; i++)
	{
		Mat view, viewGray;

		std::cout << "\n";
		if (i < (int)imageList.size())
		{
			std::cout << "  " << imageList[i];
			view = imread(imageList[i], 1);
		}

		if (view.empty())  // 若有图片数据有问题
		{
			if (imagePoints.size() > 0)
				runAndSave(outputFilename, imagePoints, imageSize,
					boardSize, pattern, squareSize, aspectRatio,
					flags, cameraMatrix, distCoeffs,
					writeExtrinsics, writePoints);
			break;
		}

		imageSize = view.size();

		if (flipVertical)	flip(view, view, 0);

		vector<Point2f> pointbuf;
		cvtColor(view, viewGray, COLOR_BGR2GRAY);

		bool found = findChessboardCorners(view, boardSize, pointbuf,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

		if (pattern == CHESSBOARD && found)
			cornerSubPix(viewGray, pointbuf, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

		if (mode == CAPTURING && found)
		{
			imagePoints.push_back(pointbuf);
			cout << ", Success";
			sucessImgs << imageList[i] << " successed." << endl;
		}
		else
			sucessImgs << imageList[i] << " failed." << endl;

		if (found)	drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

		string msg = mode == CAPTURING ? "100/100" : mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		putText(view, format("%d/%d", (int)imagePoints.size(), nframes),
			textOrigin, 1, 1,
			mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

		imshow("Image View", view);

		if (waitKey(500) == 27)	break;

		if (mode == CAPTURING && imagePoints.size() >= (unsigned)nframes)
		{
			if (runAndSave(outputFilename, imagePoints, imageSize,
				boardSize, pattern, squareSize, aspectRatio,
				flags, cameraMatrix, distCoeffs,
				writeExtrinsics, writePoints))
				mode = CALIBRATED;
			else
				mode = DETECTION;
		}
	}


	// 显示
	if (showUndistorted)
	{
		int n = 1;  // Undistort
		string funcName[] = { "Undistort", "Remap" };

		Mat view, rview, map1, map2;

		if (n == 1) {
			initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
				getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
				imageSize, CV_16SC2, map1, map2);
		}

		for (int i = 0; i < (int)imageList.size(); i++)
		{
			view = imread(imageList[i], 1);
			if (view.empty())	continue;

			if (n == 0)			undistort(view, rview, cameraMatrix, distCoeffs, cameraMatrix);
			else if (n == 1)	remap(view, rview, map1, map2, INTER_LINEAR);

			imshow("Image View " + funcName[n], rview);
			char c = (char)waitKey(500);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	cv::destroyAllWindows();

	::system("pause");
	return 0;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void Tsai_HandEye(Mat& Hcg, vector<Mat>& Hgij, vector<Mat>& Hcij)
{
	CV_Assert(Hgij.size() == Hcij.size());
	int nStatus = Hgij.size();

	Mat Rgij(3, 3, CV_64FC1);
	Mat Rcij(3, 3, CV_64FC1);

	Mat rgij(3, 1, CV_64FC1);
	Mat rcij(3, 1, CV_64FC1);

	double theta_gij;
	double theta_cij;

	Mat rngij(3, 1, CV_64FC1);
	Mat rncij(3, 1, CV_64FC1);

	Mat Pgij(3, 1, CV_64FC1);
	Mat Pcij(3, 1, CV_64FC1);

	Mat tempA(3, 3, CV_64FC1);
	Mat tempb(3, 1, CV_64FC1);

	Mat A;
	Mat b;
	Mat pinA;

	Mat Pcg_prime(3, 1, CV_64FC1);
	Mat Pcg(3, 1, CV_64FC1);
	Mat PcgTrs(1, 3, CV_64FC1);

	Mat Rcg(3, 3, CV_64FC1);
	Mat eyeM = Mat::eye(3, 3, CV_64FC1);

	Mat Tgij(3, 1, CV_64FC1);
	Mat Tcij(3, 1, CV_64FC1);

	Mat tempAA(3, 3, CV_64FC1);
	Mat tempbb(3, 1, CV_64FC1);

	Mat AA;
	Mat bb;
	Mat pinAA;

	Mat Tcg(3, 1, CV_64FC1);

	auto skew = [](cv::Mat mp) ->cv::Mat {
		// ......

		cv::Mat tmp = Mat::zeros(cv::Size{ 3,3 }, CV_64FC1);
		tmp.at<double>(0, 1) = -mp.at<double>(2, 0); tmp.at<double>(1, 0) = mp.at<double>(2, 0);
		tmp.at<double>(0, 2) = mp.at<double>(1, 0); tmp.at<double>(2, 0) = -mp.at<double>(1, 0);
		tmp.at<double>(1, 2) = -mp.at<double>(0, 0); tmp.at<double>(2, 1) = mp.at<double>(0, 0);
		return tmp;
	};


	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
		Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);

		Rodrigues(Rgij, rgij);
		Rodrigues(Rcij, rcij);

		theta_gij = norm(rgij);
		theta_cij = norm(rcij);

		rngij = rgij / theta_gij;
		rncij = rcij / theta_cij;

		Pgij = 2 * sin(theta_gij / 2)*rngij;
		Pcij = 2 * sin(theta_cij / 2)*rncij;

		tempA = skew(Pgij + Pcij);
		tempb = Pcij - Pgij;

		A.push_back(tempA);
		b.push_back(tempb);
	}

	//Compute rotation
	invert(A, pinA, DECOMP_SVD);

	Pcg_prime = pinA * b;
	Pcg = 2 * Pcg_prime / sqrt(1 + norm(Pcg_prime) * norm(Pcg_prime));
	PcgTrs = Pcg.t();
	Rcg = (1 - norm(Pcg) * norm(Pcg) / 2) * eyeM + 0.5 * (Pcg * PcgTrs + sqrt(4 - norm(Pcg)*norm(Pcg))*skew(Pcg));

	//Computer Translation 
	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
		Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);
		Hgij[i](Rect(3, 0, 1, 3)).copyTo(Tgij);
		Hcij[i](Rect(3, 0, 1, 3)).copyTo(Tcij);


		tempAA = Rgij - eyeM;
		tempbb = Rcg * Tcij - Tgij;

		AA.push_back(tempAA);
		bb.push_back(tempbb);
	}

	invert(AA, pinAA, DECOMP_SVD);
	Tcg = pinAA * bb;

	Rcg.copyTo(Hcg(Rect(0, 0, 3, 3)));
	Tcg.copyTo(Hcg(Rect(3, 0, 1, 3)));
	Hcg.at<double>(3, 0) = 0.0;
	Hcg.at<double>(3, 1) = 0.0;
	Hcg.at<double>(3, 2) = 0.0;
	Hcg.at<double>(3, 3) = 1.0;
}

void Navy_HandEye(Mat& Hcg, vector<Mat>& Hgij, vector<Mat>& Hcij)
{
	CV_Assert(Hgij.size() == Hcij.size());
	int nStatus = Hgij.size();

	Mat Rgij(3, 3, CV_64FC1);
	Mat Rcij(3, 3, CV_64FC1);

	Mat alpha1(3, 1, CV_64FC1);
	Mat beta1(3, 1, CV_64FC1);
	Mat alpha2(3, 1, CV_64FC1);
	Mat beta2(3, 1, CV_64FC1);
	Mat A(3, 3, CV_64FC1);
	Mat B(3, 3, CV_64FC1);

	Mat alpha(3, 1, CV_64FC1);
	Mat beta(3, 1, CV_64FC1);
	Mat M(3, 3, CV_64FC1, Scalar(0));

	Mat MtM(3, 3, CV_64FC1);
	Mat veMtM(3, 3, CV_64FC1);
	Mat vaMtM(3, 1, CV_64FC1);
	Mat pvaM(3, 3, CV_64FC1, Scalar(0));

	Mat Rx(3, 3, CV_64FC1);

	Mat Tgij(3, 1, CV_64FC1);
	Mat Tcij(3, 1, CV_64FC1);

	Mat eyeM = Mat::eye(3, 3, CV_64FC1);

	Mat tempCC(3, 3, CV_64FC1);
	Mat tempdd(3, 1, CV_64FC1);

	Mat C;
	Mat d;
	Mat Tx(3, 1, CV_64FC1);

	//Compute rotation
	if (Hgij.size() == 2) // Two (Ai,Bi) pairs
	{
		Rodrigues(Hgij[0](Rect(0, 0, 3, 3)), alpha1);
		Rodrigues(Hgij[1](Rect(0, 0, 3, 3)), alpha2);
		Rodrigues(Hcij[0](Rect(0, 0, 3, 3)), beta1);
		Rodrigues(Hcij[1](Rect(0, 0, 3, 3)), beta2);

		alpha1.copyTo(A.col(0));
		alpha2.copyTo(A.col(1));
		(alpha1.cross(alpha2)).copyTo(A.col(2));

		beta1.copyTo(B.col(0));
		beta2.copyTo(B.col(1));
		(beta1.cross(beta2)).copyTo(B.col(2));

		Rx = A*B.inv();

	}
	else // More than two (Ai,Bi) pairs
	{
		for (int i = 0; i < nStatus; i++)
		{
			Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
			Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);

			Rodrigues(Rgij, alpha);
			Rodrigues(Rcij, beta);

			M = M + beta*alpha.t();
		}

		MtM = M.t()*M;
		eigen(MtM, vaMtM, veMtM);

		pvaM.at<double>(0, 0) = 1 / sqrt(vaMtM.at<double>(0, 0));
		pvaM.at<double>(1, 1) = 1 / sqrt(vaMtM.at<double>(1, 0));
		pvaM.at<double>(2, 2) = 1 / sqrt(vaMtM.at<double>(2, 0));

		Rx = veMtM*pvaM*veMtM.inv()*M.t();
	}

	//Computer Translation 
	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
		Hgij[i](Rect(3, 0, 1, 3)).copyTo(Tgij);

		Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);
		Hcij[i](Rect(3, 0, 1, 3)).copyTo(Tcij);

		tempCC = eyeM - Rgij;
		tempdd = Tgij - Rx * Tcij;

		C.push_back(tempCC);
		d.push_back(tempdd);
	}

	Tx = (C.t()*C).inv()*(C.t()*d);

	Hcg.create(cv::Size(4, 4), CV_64F);
	Rx.copyTo(Hcg(Rect(0, 0, 3, 3)));
	Tx.copyTo(Hcg(Rect(3, 0, 1, 3)));
	Hcg.at<double>(3, 0) = 0.0;
	Hcg.at<double>(3, 1) = 0.0;
	Hcg.at<double>(3, 2) = 0.0;
	Hcg.at<double>(3, 3) = 1.0;

}

// 仅用前三个处理
void Navy_HandEye_Easy(Mat& Hcg, vector<Mat>& Hgij, vector<Mat>& Hcij)
{
	// Rodrigues(Hgij[0](Rect(0, 0, 3, 3)), alpha1);

}


static bool isRotationMatrix(Mat &R)
{
	Mat Rt;
	transpose(R, Rt);
	Mat shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

	return  norm(I, shouldBeIdentity) < 1e-6;
}


static void rotMatZYZ(Mat& eulerAngle, Mat& rotMat) {
	CV_Assert(eulerAngle.rows ==1 && eulerAngle.cols == 3);
	Matx13d m(eulerAngle);
	auto rx = m(0, 0), ry = m(0, 1), rz = m(0, 2);
	auto xs = std::sin(rx), xc = std::cos(rx);
	auto ys = std::sin(ry), yc = std::cos(ry);
	auto zs = std::sin(rz), zc = std::cos(rz);
	
	Mat rotX = (Mat_<double>(3, 3) << 1, 0, 0, 0, xc, -xs, 0, xs, xc);
	Mat rotY = (Mat_<double>(3, 3) << yc, 0, ys, 0, 1, 0, -ys, 0, yc);
	Mat rotZ = (Mat_<double>(3, 3) << zc, -zs, 0, zs, zc, 0, 0, 0, 1);

	rotMat =  rotX*rotY*rotZ;

	//cout << isRotationMatrix(rotMat) << endl;
}


//  6元素 转 姿态矩阵
/*void*/ cv::Mat sixElementsToMatrix(cv::Mat& m, bool default = true)
{
	CV_Assert(m.total() == 6);
	if (m.cols == 1)
		m = m.t();

	Mat tmp = Mat::eye(4, 4, CV_64FC1);

	if (default)
		cv::Rodrigues(m({ 3, 0, 3, 1 }), tmp({ 0, 0, 3, 3 }));
	else
		rotMatZYZ(m({ 3, 0, 3, 1 }), tmp({ 0, 0, 3, 3 }));

	tmp({ 3, 0, 1, 3 }) = m({ 0, 0, 3, 1 }).t();
	//std::swap(m,tmp);
	return tmp;
}


// 数据表示均为 (x,y,x,rx,ry,rz)
// 返回数据为 4*4 的旋转矩阵 向量
bool readDatasFromFile(std::string file, std::vector<cv::Mat> & vecHg, std::vector<cv::Mat> & vecHc)
{
	ifstream infile(file, ios::in);
	if (!infile.is_open()) {
		std::cout << " file path error." << std::endl; return false;
	}

	std::vector<cv::Mat> linesDatas;
	while (!infile.eof()) {
		Mat temp(1, 6, CV_64FC1);
		for (int i = 0; i < 6; ++i)
			infile >> temp.at<double>(0, i);
		//cout << temp << endl;
		linesDatas.emplace_back(std::move(temp));
	}

	size_t sz = linesDatas.size();
	if (sz % 2 == 1) {
		std::cout << " data items number not match." << std::endl; return false;
	}

	vecHg.assign(linesDatas.cbegin(), linesDatas.cbegin() + sz / 2);
	vecHc.assign(linesDatas.cbegin() + sz / 2, linesDatas.cend());

	// angle to rad
	for (auto& m : vecHg)
		m({ 3,0,3,1 }) *= CV_PI / 180.;

	/////////////////////////////////////////
	// vector to matrix
	for (auto& m : vecHg) 
		m = sixElementsToMatrix(m,false);  // 欧拉角->旋转矩阵

	for (auto& m : vecHc)
		m = sixElementsToMatrix(m);		// 旋转向量->旋转矩阵
}

void convertVectors2Hij(std::vector<cv::Mat> & vecHg, std::vector<cv::Mat> & vecHc,
	std::vector<cv::Mat> & vecHgij, std::vector<cv::Mat> & vecHcij)
{
	//H_t1 = H*H_t2
	for (int i = 0; i < vecHc.size() - 1; ++i) {
		vecHgij.emplace_back(vecHg[i].inv() * vecHg[i + 1]);
		vecHcij.emplace_back(vecHc[i] * vecHc[i + 1].inv());
	}

	cout << vecHcij[0] << endl << vecHcij[1] << endl << endl;
	cout << vecHgij[0] << endl << vecHgij[1] << endl << endl;

	vecHgij.assign(vecHg.cbegin(), vecHg.cend());
	vecHcij.assign(vecHc.cbegin(), vecHc.cend());
}

// 图像坐标（给定高度），手眼矩阵，机械手臂末端姿态矩阵，外参，内参
cv::Mat solveLocation(cv::Vec3d pos, cv::Mat& Hcg, cv::Mat& Hg, cv::Mat& extParamMatrix, cv::Mat& camMatrix)
{
	cv::Mat camMatrixExtend;
	cv::hconcat(camMatrix, cv::Mat::zeros(3, 1, CV_64F), camMatrixExtend);

	//Mat M = camMatrixExtend*Hcg*Hg.inv();   // size 3*4

	Mat M = camMatrixExtend*extParamMatrix*Hcg*Hg; 

	// Mat M = camMatrixExtend*extParamMatrix*Hcg;

	/*      { u }
	*     z { v } = M * {x,y,z,1}'
	*       { 1 }
	*/

	double  u = pos[0], v = pos[1], z = pos[2];

	Mat res;

	// solove  M * {x,y,z, 1}' = z{u,v,1}'
	//
	//  1、  ==>  M33 *{ x,y,z} = {zu - M14, zv - M24, z - M34}	
	//Mat b(3, 1, CV_64F);
	//b.at<double>(0, 0) = z*u - M.at <double>(0, 3);
	//b.at<double>(1, 0) = z*v - M.at <double>(1, 3);
	//b.at<double>(2, 0) = z   - M.at <double>(2, 3);
	//res = M({0,0,3,3}).inv()*b;  

	//  2、  ==>  M1*{x,y}'=Mr
	Matx34d MM(M);
	Mat M1 = (Mat_<double>(2, 2) <<
		u*MM(2, 0) - MM(0, 0), u*MM(2, 1) - MM(0, 1),
		v*MM(2, 0) - MM(1, 0), v*MM(2, 1) - MM(1, 1));

	Mat Mr = (Mat_<double>(2, 1) <<
		MM(0, 3) - u*MM(2, 3) - z*(u*MM(2, 2) - MM(0, 2)),
		MM(1, 3) - v*MM(2, 3) - z*(v*MM(2, 2) - MM(1, 2)));

	res = Mat(3, 1, CV_64F, cv::Scalar(z));
	res({ 0,0,1,2 }) = M1.inv()*Mr;

	return res;
}

//////////////////////////////////////////////////////////////////////////


// 内参矩阵
cv::Mat camMatrix = (cv::Mat_<double>(3, 3) <<
	3478.5165596966, 0, 1312.18749271937,
	0, 3478.50092392125, 1071.57326781329,
	0, 0, 1);

// 畸变系数
cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
	-0.0739402553270263, 0.197458868521729, 0.00153283624674466,
	0.0000894537118846742, 0.0669888572402587);


void test3()
{
	// 第一幅图 测试     // 棋盘空间坐标点 -->  图像坐标点 
	cv::Point3d objPos;
	objPos = { 0., 0., 0. };   // 棋盘坐标系
	objPos = { 0.0331, -0.06248,-0.0416 };    // 这个点是反算出来的。不断尝试不同x,y的取值，通过projectPoints映射到(1680,1630)
	//
	vector<Point3d> objPostions{ objPos };
	vector<Point2d> imgPostions;

	Mat rvec = (Mat_<double>(1, 3) << 0.464056514, 0.292694816, 2.37106839);   // 第一幅图 旋转向量
	Mat tvec = (Mat_<double>(1, 3) << 0.042598413, -0.004392188, 0.425699383); // 第一幅图 位移向量

	cv::projectPoints(Mat(objPostions), rvec, tvec, camMatrix, distCoeffs, imgPostions);    // 棋盘坐标点 -->  图像坐标点

	cout << imgPostions[0] << endl;    //  (0,0,0) -> (1660,1035) ,  (0.0331, -0.06248,-0.0416) -> (1680,1630)

	
									   
	////  测试  “棋盘世界坐标系”转换到“图像坐标系 ”    (未考虑畸变)
	//   z*{u,v,1}' = A * {R,r} * {x,y,z}'

	cv::Mat cheesePos{ 0.0331, -0.06248,-0.0416 };

	cv::Mat extParamsMatrix = 
		sixElementsToMatrix(Mat{ 0.042598413,-0.004392188,0.425699383,0.464056514,0.292694816,2.37106839 }); // 第一幅图，外参矩阵   

	Mat camPos = extParamsMatrix*Mat{ 0.0331, -0.06248, -0.0416 ,1. };  // 棋盘->相机   	

	Mat imgPos = camMatrix* camPos({ 0,0,1,3 });  // 相机 -> 图像

	imgPos /= imgPos.at<double>(2, 0);   // 除以z,保留 x，y

	cout << "\ncam pos point: " << camPos.t() << endl;
	cout << "chess pt convert to img pt: " << cheesePos.t() << " ==> "<< imgPos({ 0,0,1,2 }).t() << endl;

	cv::Mat camMatrixExtend;
	cv::hconcat(camMatrix, cv::Mat::zeros(3, 1, CV_64F), camMatrixExtend);
	//imgPos = camMatrixExtend*extParamsMatrix*Mat{ 0.0331, -0.06248, -0.0416 ,1. };
	//imgPos /= imgPos.at<double>(2, 0);
	//cout << imgPos({ 0,0,1,2 }).t() << endl;


	//////////////// 图像坐标系转棋盘世界坐标系

	//---1
	cv::Mat M = camMatrixExtend*extParamsMatrix;
	cheesePos = M.inv(cv::DECOMP_SVD)*(-0.0416*Mat{ 1680., 1630., 1. });// 反算结果不正确？？？
	cout << cheesePos/(cheesePos.at<double>(2,0)/-0.0416) << endl;

	//---2
	//cv::Mat camInvProdZImgPos = camMatrix.inv()*(-0.0416*Mat{ 1680., 1630., 1. });
	//camInvProdZImgPos.push_back(1.);
	//cheesePos = extParamsMatrix.inv()*camInvProdZImgPos;
	//cout << cheesePos({ 0,0,1,3 }) << endl;

	//---3
	//Mat M = camMatrixExtend*extParamsMatrix;
	//double  u = 1680, v = 1630, z = -0.0416;
	//Mat b(3, 1, CV_64F);
	//b.at<double>(0, 0) = z*u - M.at <double>(0, 3);
	//b.at<double>(1, 0) = z*v - M.at <double>(1, 3);
	//b.at<double>(2, 0) = z - M.at <double>(2, 3);
	//Mat res = M({0,0,3,3}).inv()*b;
	//cout << res << endl;

	//---4
	//Mat M = camMatrixExtend*extParamsMatrix;
	//double  u = 1680, v = 1630, z = -0.0416;
	//Matx34d MM(M);
	//Mat M1 = (Mat_<double>(2, 2) <<
	//	u*MM(2, 0) - MM(0, 0), u*MM(2, 1) - MM(0, 1),
	//	v*MM(2, 0) - MM(1, 0), v*MM(2, 1) - MM(1, 1));

	//Mat Mr = (Mat_<double>(2, 1) <<
	//	MM(0, 3) - u*MM(2, 3) - z*(u*MM(2, 2) - MM(0, 2)),
	//	MM(1, 3) - v*MM(2, 3) - z*(v*MM(2, 2) - MM(1, 2)));

	//Mat res = M1.inv()*Mr;
	//cout << res << endl;



}


void test2()
{
	std::vector<cv::Mat> vecHg, vecHc;
	readDatasFromFile("cam_chess_datas.txt", vecHg, vecHc);

	//Mat tmp1, tmp2;
	//Mat rotMat(vecHg[0]({ 0, 0, 3, 3 }));
	//Rodrigues(rotMat, tmp1);
	//Rodrigues(tmp1, tmp2);   // 2次转换后，旋转矩阵是一样的， 旋转向量有变化（多个旋转向量对应一个旋转矩阵）

	Mat Hcg;

	Navy_HandEye(Hcg, vecHc, vecHg);

	//////////////////////////////////////////////////////////////////

	// test pixel 2d point to wordl 3d point
	cv::Mat extParamsMatrix = vecHc[0];  // 外参矩阵
	cv::Mat Hg = vecHg[0]; // 姿态矩阵

	/*
		double z = -0.0416;
		cv::Vec3d pos = { 1680,1630, 1 };  // 相机世界点 

		Mat pPix(pos);						// 像素坐标（齐次）

		Mat pCam = camMatrix.inv()*z*pPix;	// 相机坐标系
		Mat pCamT(4, 1, CV_64F, Scalar(1)); pCam.copyTo(pCamT({ 0,0,1,3 }));

		Mat pW = extParamsMatrix.inv()*pCamT; // 世界坐标系

		Mat pT = Hcg.inv()*pCamT;   // 
	*/
	cv::Mat camPos{ 0.03977475914407023, 0.06037057858452912, 0.3754228333648103, 1. };  // 相机世界点 

	cv::Mat endPos = Hcg.inv()*camPos;   // 

	//endPos = camPos;

	cv::Mat worldPos = Hg*endPos;


}

void test1()
{
	// 标定相机
	//calibration();


	// 读取 预处理好的txt  前一半 手臂姿态 ， 后一半 外参
	std::vector<cv::Mat> vecHg, vecHc;
	readDatasFromFile("cam_chess_datas.txt", vecHg, vecHc);


	// 手眼标定
	Mat Hcg;
	std::vector<cv::Mat> vecHgij, vecHcij;
	convertVectors2Hij(vecHg, vecHc, vecHgij, vecHcij);

	Navy_HandEye(Hcg, vecHgij, vecHcij);
	//Navy_HandEye(Hcg, vector<Mat>{ vecHgij[0], vecHgij[1] }, vector<Mat>{ vecHcij[0],vecHc[1] });

	cout << "X = " << endl << Hcg << endl;


	// test pixel 2d point to wordl 3d point
	cv::Mat extParamsMatrix = vecHc[0];
	cv::Mat Hg = vecHg[0];

	double z = -0.0416;
	cv::Vec3d pos = { 1680,1630, 1 };
	Mat loc = solveLocation(pos.mul({ 1,1,z }), Hcg, Hg, extParamsMatrix, camMatrix);

}



int main() {

	test3();	
	//test2();
	//test1();

	system("pause");
}