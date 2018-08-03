#include "Calibration.h"

using namespace cv;
using namespace std;

Calibration::Calibration(const std::string& imgsDirector, 
	const std::string& outputFilename, 
	Size boardSize, 
	double squareSize)
{
	this->imgsDirectory = imgsDirector;
	this->outputFilename = outputFilename;
	this->boardSize = boardSize;
	this->squareSize = squareSize;
}

double Calibration::computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

void Calibration::calcChessboardCorners(Size boardSize, float squareSize, 
	vector<Point3f>& corners, Pattern patternType)
{
    corners.resize(0);

    switch(patternType)
    {
      case CHESSBOARD:
      case CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float(j*squareSize), 
					float(i*squareSize), 0));
        break;

      case ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize),
                                          float(i*squareSize), 0));
        break;

      default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

bool Calibration::runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize, Pattern patternType,
                    float squareSize, float aspectRatio,
                    int flags, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( flags & CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                    distCoeffs, rvecs, tvecs, flags|CALIB_FIX_K4|CALIB_FIX_K5);
                    ///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


void Calibration::saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );
	if (!fs.isOpened()) {
		std::cout << "Can not create \"" << filename << "\"." << std::endl;	return;
	}

	if (!cv::utils::fs::exists(filename))
		cv::error(cv::Error::StsBadArg, "file create failed.",__FUNCTION__,__FILE__,__LINE__);

    time_t tt;
    time( &tt );
	struct tm *t2 = new struct tm();
	localtime_s(t2, &tt);
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        sprintf_s( buf, "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	//// 增加旋转、平移
	//fs << "rvecs" << rvecs;
	//fs << "distortion_coefficients" << distCoeffs;

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
			// {r, t}
            Mat r = bigmat(Range(i, i+1), Range(0,3));    
            Mat t = bigmat(Range(i, i+1), Range(3,6));

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

		std::swap(bigmat, extrinsicsBigMat);
    }

	/////  0731 存储检测棋盘结果  1  成功， 0  失败  ==》 对应1的菜读取机械手姿态数据
	Mat matFoundCheeseBoard(1, foundCheeseBoardVec.size(), CV_32S, foundCheeseBoardVec.data());
	fs << "found_cheese_board" << matFoundCheeseBoard;


    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
	fs.release();

	std::cout << "Calibration done, save datas in file " << filename << endl << endl;
}

bool Calibration::runAndSave(const string& outputFilename,
                const vector<vector<Point2f> >& imagePoints,
                Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                float aspectRatio, int flags, Mat& cameraMatrix,
                Mat& distCoeffs, bool writeExtrinsics, bool writePoints )
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

    if( ok )
        saveCameraParams( outputFilename, imageSize,
                         boardSize, squareSize, aspectRatio,
                         flags, cameraMatrix, distCoeffs,
                         writeExtrinsics ? rvecs : vector<Mat>(),
                         writeExtrinsics ? tvecs : vector<Mat>(),
                         writeExtrinsics ? reprojErrs : vector<float>(),
                         writePoints ? imagePoints : vector<vector<Point2f> >(),
                         totalAvgErr );
    return ok;
}

bool Calibration::doCalibration()
{
	////// load images 
	vector<String> imageList;
	std::string path = this->imgsDirectory;

	cv::glob(path + "/*.bmp", imageList);  // assume no exits sequence: 1,2,..,10,11,..,20,21
											// will get 1,10,11,12,...,19,2,20, 21
	if (imageList.size() == 0) {
		std::cout << "no images." << std::endl;	return false;
	}

	int nframes = (int)imageList.size();

	this->foundCheeseBoardVec.resize(nframes);

	if (showUndistorted) {
		namedWindow("Image View", 1);
		//resizeWindow("Image View", { 800,800 });
	}

	std::cout << "Process: ";
	for (int i = 0;; i++)
	{
		Mat view, viewGray;

		std::cout << "\n";
		if (i < (int)imageList.size())	{
			std::cout << "  " << imageList[i];
			view = imread(imageList[i], 1);
		}

		if (view.empty())  // use previous imgs to calibration
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

		if (mode == CAPTURING && found)	{
			imagePoints.push_back(pointbuf);
			cout << ", Success";
			foundCheeseBoardVec[i] = 1;
		}
		else {
			foundCheeseBoardVec[i] = 0;
		}

		if (found)	drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

		string msg = mode == CAPTURING ? "100/100" : mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		putText(view, format("%d/%d", (int)imagePoints.size(), nframes),
			textOrigin, 1, 1,
			mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

		imshow("Image View", view);

		if (waitKey(300) == 27)	break;

		if (mode == CAPTURING && imagePoints.size() > (unsigned)nframes)
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

	/// show undistorted imgs
	if (showUndistorted) // default: false
	{
		int n = 1;  // Undistort
		string funcName[] = { "Undistort", "Remap" };
		Mat view, rview;
		for (int i = 0; i < (int)imageList.size(); i++)		{
			view = imread(imageList[i], 1);
			if (view.empty())	continue;
			if (n == 0)			
				undistort(view, rview, cameraMatrix, distCoeffs, cameraMatrix);
			else if (n == 1) {
				Mat map1, map2;
				initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
					getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
					imageSize, CV_16SC2, map1, map2);
				remap(view, rview, map1, map2, INTER_LINEAR);
			}
			imshow("Image View " + funcName[n], rview);
			char c = (char)waitKey(300);
			if (c == 27 || c == 'q' || c == 'Q')break;
		}
	}
	cv::destroyAllWindows();
    return true;
}


// Get Method
cv::Mat Calibration::getExtrinsicsBigMat() const
{
	return this->extrinsicsBigMat;
}

cv::Mat Calibration::getCameraMatrix() const
{
	return this->cameraMatrix;
}

cv::Mat Calibration::getDistCoeffsMatrix() const
{
	return this->distCoeffs;
}

vector<int> Calibration::getFoundCheeseBoardVec() const
{
	return this->foundCheeseBoardVec;
}