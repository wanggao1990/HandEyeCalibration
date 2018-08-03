#include "HandEyeCalibration.h"


int main()
{	
	std::vector<int> vecInt(10, 1);
	std::vector<int> vecInt2(1);



	
	// 标定 
	//if(true){
	//	std::string path = R"(C:\Users\wanggao\Desktop\eyeinhand\图片0731)";
	//	HandEyeCalibration handEyeTest(path, "out_camera_data.yml",{7,6},25);
	//	
	//	if (handEyeTest.doCalibration())
	//	{
	//		for (auto& b : handEyeTest.getVecFoundCheeseBoard()) {
	//			std::cout << b << std::endl;
	//		}
	//	}
	//}

	//HandEyeCalibration::test3();


	//////////////////////////////////////////////////////////////////////////

	if (0)
	{
		Mat camMatrix = (Mat_<double>(3, 3) <<
			5.9554307895791999e+03, 0., 6.5223928155464080e+02,
			0., 5.9575910515278119e+03, 4.8330008738099161e+02,
			0., 0., 1.);

		Mat distCoefs{
			-1.8372222355673893e+00, 1.1349449068343554e+02,
			-4.5780638529011730e-04, -8.5438069699633192e-04,6.8780241055107350e-01
		};

		Mat camExt1{
			6.6174876437763643e+01,	9.3974558130480919e+00, 1.9895285192546758e+03,
			2.7332879540652201e-02, -2.0037203697984007e-02,1.5690394552147897e+00,
		};

		Mat camExt2{
			6.3421670844796253e+01,	1.3924721151854603e+01, 3.0927962015205499e+03,
			1.7512944721410115e-02, -4.0035476787074094e-03,1.5663633383314310e+00,
		};

		Mat camExt3{
			7.6731427825588128e+01,	6.6827237461899514e+01, 1.9905040685698120e+03,
			2.1154360398431372e-02, -4.2893642258009047e-02,2.6958709741477986e+00,
		};


		Mat camExtParam1 = HandEyeCalibration::attitudeVectorToMatrix(camExt1);
		Mat camExtParam2 = HandEyeCalibration::attitudeVectorToMatrix(camExt2);
		Mat camExtParam3 = HandEyeCalibration::attitudeVectorToMatrix(camExt3);

		Mat ttt = HandEyeCalibration::attitudeVectorToMatrix(Mat{ -2.58,  11.17 , 1597.97, -2.21, -2.20, -0.03 });

		//Mat camPos1 = /*camMatrix*Mat::eye(3,4,CV_64F)**/camExtParam1*Mat{0.,80., 0., 1.};
		//Mat camPos2 = camExtParam2*Mat{ 0., 80., 0., 1. };

		//cout << camPos1 << endl;
		//cout << camPos2 << endl;

		// 
		Mat imgPos{ 640.,512. };
		Mat cheesePos{ 0.,70.,0., 1. };

		Mat camPos1 = camExtParam1*cheesePos;
		Mat camPos2 = camExtParam2*cheesePos;
		Mat camPos3 = camExtParam3*cheesePos;

		std::cout << camPos1 << endl << camPos2 << endl << norm(camPos1 - camPos2) << endl;


		if (0) 
		{
			Point2d imgPos{ 640.,512. };
			vector<Point3d> objPostions{ {0.,70.,0.} };

			Mat ext1t = camExt1({ 0, 0, 3, 1 }); // 前三个为t
			Mat ext1R = camExt1({ 3, 0, 3, 1 }); // 后三个为r

			double propZc;
			double minDis = 500;
			for (double zc = 500; zc < 2200; zc += 5)
			{
				Mat imgP = camMatrix*Mat::eye(3, 4, CV_64F)*Mat { 0., 70. / zc, 0., 1. };
				imgP /= imgP.at<double>(2, 0);
				double dis = cv::norm(Mat(imgP({ 0,0,1,2 })), Mat(imgPos), cv::NORM_L2);


				//Mat imgP = camMatrix*(camExtParam1*Mat{ 0., 70. / zc, 0., 1. })({0,0,1,3});
				//imgP /= imgP.at<double>(2, 0);
				//double dis = cv::norm(Mat(imgP({0,0,1,2})), Mat(imgPos), cv::NORM_L2);

				//vector<Point2d> imgP;
				//cv::projectPoints(Mat(objPostions), ext1R, ext1t, camMatrix, distCoefs, imgP);
				//double dis = cv::norm(Mat(imgP[0]), Mat(imgPos),cv::NORM_L2);

				if (dis < minDis)
				{
					minDis = dis;
					propZc = zc;
					std::cout << propZc << " " << minDis << endl;
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	

	if (0)
	{
		// 内参
		Mat camMatrix, distCoefs;
		HandEyeCalibration::readCameraParameters("out_camera_data.yml", camMatrix, distCoefs);

		// 外参， 机械手姿态
		std::vector<cv::Mat> vecHg, vecHc;
		HandEyeCalibration::readDatasFromFile("cam_chess_datas.txt", vecHg, vecHc);

		// 求解，手眼标定矩阵
		Mat Hcg;
		std::vector<cv::Mat> vecHgij, vecHcij;
		HandEyeCalibration::convertVectors2Hij(vecHg, vecHc, vecHgij, vecHcij);
		
		//HandEyeCalibration::Navy_HandEye(Hcg, vecHgij, vecHcij);
		HandEyeCalibration::Navy_HandEye(Hcg, vecHgij, vecHcij);

		//////////////////////////////////////////////////////////////////////////

		//for (int i=0;i<15;i++)
		//{
		//	Mat res = vecHc[i] * Hcg*vecHg[i].inv();
		//	cout << res << endl;
		//}

		// 判断是否旋转矩阵 
		//Mat R = vecHc[0]({ 0,0,3,3 });
		//cout << HandEyeCalibration::isRotationMatrix(R) << endl;
		//cout << R << endl << R.inv() << endl << R.t() << endl;

		//// 两个旋转矩阵，求解变换矩阵也是旋转矩阵，直接求和分开求结果一致
		// {R1, t1}*{R,t}={R2,t2} 直接求和分开求结果一致


		Mat r = HandEyeCalibration::quaternionToRotatedMatrix({ 0.707,0.,0.,0.707 });
		cout << r<< endl << HandEyeCalibration::isRotationMatrix(r) << endl;
		
		//////////////////////////////////////////////////////////////////////////
		
		// 指定高度下，图像中点对应的空间坐标
		auto res = HandEyeCalibration::getWorldPos(Point2d(1680, 1630), -0.0416,
			Hcg, vecHg[0], camMatrix);
		cout << res << endl;

		//////////////////////////////////////////////////////////////////////////

		//Hcg = (Mat_<double>(4, 4) <<
		//	0.1300, 0.9608, -0.2212, -0.1800,
		//	0.9943, -0.1015, 0.0816, 0.3365,
		//	0.0545, -0.2274, -0.9684, 0.5768,
		//	0, 0, 0, 1.0000
		//	);  // matlab求解结果

		// 根据空间点 计算图像坐标

		for (auto& Hg : vecHg)
		{
			Mat hPos = Hg * Mat{ -0.2874, 0.2048, -0.0416, 1. };

			Mat camPos = Hcg.inv()*hPos;
			Mat pixelPos = camMatrix*Mat::eye(3, 4, CV_64F) * camPos;
			pixelPos /= pixelPos.at<double>(2, 0);
			cout << pixelPos.t() << endl;
		}
		
	}


	if (1)  // 0731 测试
	{
		////// 标定
		//if (true) {
		//	std::string path = R"(C:\Users\wanggao\Desktop\eyeinhand\图片0731)";
		//	HandEyeCalibration handEyeTest(path, "out_camera_data.yml", { 7,6 }, 0.025);
		//	bool ret = handEyeTest.doCalibration();
		//	if (ret)
		//	{
		//		auto camMat = handEyeTest.getCameraMatrix();
		//		auto distCoffMat = handEyeTest.getDistCoeffsMatrix();
		//		auto extBigMat = handEyeTest.getExtrinsicsBigMat();
		//		auto hasCheessVec = handEyeTest.getFoundCheeseBoardVec();
		//		for (auto& v : hasCheessVec)	
		//			cout << v << " ";
		//		cout << endl;
		//		cout << endl << camMat      << endl;
		//		cout << endl << distCoffMat << endl;
		//		cout << endl << extBigMat   << endl << endl;  // {r,t格式}
		//	}
		//}

		//////////////////////////////////////////////////////////////////////////

		//// 读取外参 姿态矩阵
		std::vector<cv::Mat> vecHg, vecHc;
		bool ret = HandEyeCalibration::readDatasFromFile("out_camera_data.yml", 
			R"(C:\Users\wanggao\Desktop\eyeinhand\图片0731\pose)",	vecHg, vecHc, true	);

		//// 手眼标定
		// 求An,Bn
		std::vector<cv::Mat> vecHgij, vecHcij;
		HandEyeCalibration::convertVectors2Hij(vecHg, vecHc, vecHgij, vecHcij);
		// 求X
		cv::Mat Hcg;
		HandEyeCalibration::computerHandEyeMatrix(Hcg, vecHgij, vecHcij, 0);  // Tsai
		cout << Hcg << endl  << HandEyeCalibration::isRotationMatrix(Hcg) << endl << endl;

		//  hg0, hcg, hc0, hg1 => hc1
		cout << vecHg[0] * Hcg.inv()*vecHc[0] << endl << vecHg[1]  * Hcg.inv() * vecHc[1] << endl << endl;

		cout << vecHc[1] << endl;
		cout << Hcg* vecHg[1].inv()* vecHg[0] * Hcg.inv()*vecHc[0] << endl << endl;

		//////////////////////////////////////////////////////////////////////////
		
		Mat camMatrix, distCoefs;
		ret = HandEyeCalibration::readCameraParameters("out_camera_data.yml", camMatrix, distCoefs);

		// img pos -> world pos
		Mat imgPt = Mat{ 1446., 1154., 1. };
		double z = vecHc[0].at<double>(2, 3);
		Mat tm = camMatrix.inv(cv::DECOMP_SVD);
		tm *= imgPt;
		tm *= z;
		Mat ttm{ 1.,1.,1.,1. };  // CAUTION!!!
		tm.copyTo(ttm({ 0,0,1,3 }));
		Mat r = vecHg[0] * Hcg.inv()*ttm;
		cout << r.t()<< endl ;

		//////////////////////////////////////////////////////////////////////////
		cv::FileStorage fs("HandEyeMatrix.yml", cv::FileStorage::READ);
		Mat X = fs["handEyeMatrix"].mat();
		cv::Point3d pos= HandEyeCalibration::getWorldPos(
			cv::Point2d(Mat{ 1446., 1154. }), z, X, vecHg[0], camMatrix, vecHg[1], vecHc[1]);
		cout << pos << endl;

		//////////////////////////////////////////////////////////////////////////
		// fixed cheese pos -> world pos
		for (int i=0;i < vecHc.size(); ++i)
		{
			Mat cheesePos{ 0.,0.025,0., 1. };
			Mat worldPos = vecHg[i] * Hcg.inv()*vecHc[i] * cheesePos;
			cout << "Pos " << i << ": " << worldPos.t() << endl;
		}
	}
	return 0;
}