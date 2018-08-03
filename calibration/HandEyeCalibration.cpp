#include "HandEyeCalibration.h"

HandEyeCalibration::HandEyeCalibration(const std::string& imgsDirector,
	const std::string& outputFilename,	Size boardSize,	double squareSize) :	
	Calibration(imgsDirector, outputFilename, boardSize, squareSize)
{
}

HandEyeCalibration::~HandEyeCalibration()
{
}

void HandEyeCalibration::computerHandEyeMatrix(Mat & Hcg, vector<Mat>& Hgij, vector<Mat>& Hcij, int method,
	const std::string& file)
{
	switch (method)
	{
	case 0:
		Navy_HandEye(Hcg, Hgij, Hcij);
		break;
	case 1:
		Tsai_HandEye(Hcg, Hgij, Hcij);
		break;
	default: 
		Navy_HandEye(Hcg, Hgij, Hcij);	
		break;
	}

	cv::FileStorage fs(file, cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		std::cout << "Can not write hand-eye-matrix to \"" << file << "\" failed." << std::endl;
		return;
	}
	fs << "handEyeMatrix" << Hcg;
	fs.release();
}

void HandEyeCalibration::Tsai_HandEye(Mat& Hcg, vector<Mat>& Hgij, vector<Mat>& Hcij)
{
	CV_Assert(Hgij.size() == Hcij.size());
	int nStatus = Hgij.size();

	auto skew = [](cv::Mat mp) {
		// ......
		cv::Mat tmp = Mat::zeros(cv::Size{ 3,3 }, CV_64FC1);
		tmp.at<double>(0, 1) = -mp.at<double>(2, 0);
		tmp.at<double>(1, 0) = mp.at<double>(2, 0);
		tmp.at<double>(0, 2) = mp.at<double>(1, 0);
		tmp.at<double>(2, 0) = -mp.at<double>(1, 0);
		tmp.at<double>(1, 2) = -mp.at<double>(0, 0);
		tmp.at<double>(2, 1) = mp.at<double>(0, 0);
		return tmp;
	};

	Mat Pgij, Pcij;
	Mat Rgij, Rcij;
	Mat rgij, rcij;
	Mat rngij, rncij;
	double theta_gij, theta_cij;
	Mat A, b;

	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i]({ 0, 0, 3, 3 }).copyTo(Rgij);
		Hcij[i]({ 0, 0, 3, 3 }).copyTo(Rcij);

		Rodrigues(Rgij, rgij);
		Rodrigues(Rcij, rcij);

		theta_gij = norm(rgij);
		theta_cij = norm(rcij);

		rngij = rgij / theta_gij;
		rncij = rcij / theta_cij;

		Pgij = 2 * sin(theta_gij / 2)*rngij;
		Pcij = 2 * sin(theta_cij / 2)*rncij;

		A.push_back(skew(Pgij + Pcij));
		b.push_back(Pcij - Pgij);
	}

	//Compute rotation
	Mat eyeM = Mat::eye(3, 3, CV_64FC1);
	Mat pinA = A.inv(cv::DECOMP_SVD);
	Mat Pcg_prime = pinA * b;
	Mat Pcg = 2 * Pcg_prime / sqrt(1 + norm(Pcg_prime) * norm(Pcg_prime));
	Mat PcgTrs = Pcg.t();
	Mat Rcg = (1 - norm(Pcg) * norm(Pcg) / 2) * eyeM
		+ 0.5 * (Pcg * PcgTrs + sqrt(4 - norm(Pcg)*norm(Pcg))*skew(Pcg));

	//Computer Translation 
	Mat AA, bb;
	Mat Tgij, Tcij;
	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i]({ 0, 0, 3, 3 }).copyTo(Rgij);
		Hcij[i]({ 0, 0, 3, 3 }).copyTo(Rcij);
		Hgij[i]({ 3, 0, 1, 3 }).copyTo(Tgij);
		Hcij[i]({ 3, 0, 1, 3 }).copyTo(Tcij);

		AA.push_back(Rgij - eyeM);
		bb.push_back(Rcg * Tcij - Tgij);
	}
	Mat Tcg = AA.inv(cv::DECOMP_SVD) * bb;

	Hcg = Mat::eye(4, 4, CV_64F);
	Rcg.copyTo(Hcg({ 0, 0, 3, 3 }));
	Tcg.copyTo(Hcg({ 3, 0, 1, 3 }));
}

void HandEyeCalibration::Navy_HandEye(Mat& Hcg, vector<Mat>& Hgij, vector<Mat>& Hcij)
{
	CV_Assert(Hgij.size() == Hcij.size());
	int nStatus = Hgij.size();

	Mat Rx, Tx;
	Mat Rgij, Rcij, Tgij, Tcij;

	//Compute rotation
	if (Hgij.size() == 2) // Two (Ai,Bi) pairs
	{
		Mat alpha1, beta1;
		Mat alpha2, beta2;
		Mat A(3, 3, CV_64FC1), B(3, 3, CV_64FC1);

		Rodrigues(Hgij[0]({0, 0, 3, 3}), alpha1);
		Rodrigues(Hgij[1]({0, 0, 3, 3}), alpha2);

		Rodrigues(Hcij[0]({0, 0, 3, 3}), beta1);
		Rodrigues(Hcij[1]({0, 0, 3, 3}), beta2);

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
		Mat alpha, beta;
		Mat M(3, 3, CV_64FC1, Scalar(0));

		for (int i = 0; i < nStatus; i++)
		{
			Hgij[i]({ 0, 0, 3, 3 }).copyTo(Rgij);
			Hcij[i]({ 0, 0, 3, 3 }).copyTo(Rcij);

			Rodrigues(Rgij, alpha);
			Rodrigues(Rcij, beta);

			M = M + beta*alpha.t();
		}

		Mat MtM, veMtM, vaMtM;
		MtM = M.t()*M;
		eigen(MtM, vaMtM, veMtM);

		Mat pvaM(3, 3, CV_64FC1, Scalar(0));
		pvaM.at<double>(0, 0) = 1 / sqrt(vaMtM.at<double>(0, 0));
		pvaM.at<double>(1, 1) = 1 / sqrt(vaMtM.at<double>(1, 0));
		pvaM.at<double>(2, 2) = 1 / sqrt(vaMtM.at<double>(2, 0));

		Rx = veMtM*pvaM*veMtM.inv()*M.t();
	}

	//Computer Translation 
	Mat C, d;
	Mat eyeM = Mat::eye(3, 3, CV_64FC1);
	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i]({0, 0, 3, 3}).copyTo(Rgij);
		Hgij[i]({3, 0, 1, 3}).copyTo(Tgij);
		Hcij[i]({ 0, 0, 3, 3 }).copyTo(Rcij);   // unused ?
		Hcij[i]({ 3, 0, 1, 3 }).copyTo(Tcij);

		C.push_back(eyeM - Rgij);
		d.push_back(Tgij - Rx * Tcij);
	}
	Tx = (C.t()*C).inv()*(C.t()*d);

	Hcg = Mat::eye(4,4, CV_64F);
	Rx.copyTo(Hcg({0, 0, 3, 3}));
	Tx.copyTo(Hcg({3, 0, 1, 3}));
}

void HandEyeCalibration::convertVectors2Hij(
	std::vector<cv::Mat> & vecHg, std::vector<cv::Mat> & vecHc,
	std::vector<cv::Mat> & vecHgij, std::vector<cv::Mat> & vecHcij)
{
	CV_Assert(vecHg.size() == vecHc.size());
	vecHgij.reserve(vecHg.size());
	vecHcij.reserve(vecHc.size());
	for (int i = 0; i < vecHc.size() - 1; ++i) {
		// camera: A = A2*A1.inv();   roboy:  B = B2.inv()*B1
		vecHgij.emplace_back(vecHg[i + 1].inv() * vecHg[i]);
		vecHcij.emplace_back(vecHc[i + 1] * vecHc[i].inv());
	}
	//cout << vecHcij[0] << endl << vecHcij[1] << endl << endl;
	//cout << vecHgij[0] << endl << vecHgij[1] << endl << endl;
}

// [in]   (x,y,x,rx,ry,rz)
// [out]   attitude matrix 4*4
bool HandEyeCalibration::readDatasFromFile(const std::string& file,
	std::vector<cv::Mat> & vecHg, std::vector<cv::Mat> & vecHc)
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
	infile.close();

	size_t sz = linesDatas.size();
	if (sz % 2 == 1) {
		std::cout << " data items number not match." << std::endl; return false;
	}

	vecHg.assign(linesDatas.cbegin(), linesDatas.cbegin() + sz / 2);
	vecHc.assign(linesDatas.cbegin() + sz / 2, linesDatas.cend());

	/////////////////////////////////////////
	// vector to matrix
	for (auto& m : vecHg)
		m = attitudeVectorToMatrix(m, "zyx");  // euler angles -> r matrix

	for (auto& m : vecHc)
		m = attitudeVectorToMatrix(m);		// r vec -> r matrix

	return true;
}

bool HandEyeCalibration::readDatasFromFile(
	const std::string & file, 
	const std::string & folder, 
	std::vector<cv::Mat>& vecHg, std::vector<cv::Mat>& vecHc, bool useQuaternion)
{
	///// load extrinsic datas
	cv::FileStorage fs(file, cv::FileStorage::READ);
	if (!fs.isOpened())	{
		std::cout << "open \"" << file << "\" failed." << std::endl;
		return false;
	}
	cv::Mat extBigMat = fs["extrinsic_parameters"].mat();
	cv::Mat foundCheeseBoardMat = fs["found_cheese_board"].mat();   // int ,row vector
	fs.release();

	// {rx,ry,rx,x,y,z} => {x,y,z,rx,ry,rz}
	Mat tExtM = extBigMat(Range::all(), Range(3, 6)).clone();
	Mat rExtM = extBigMat(Range::all(), Range(0, 3)).clone();
	hconcat(tExtM, rExtM, extBigMat);
	
	if (extBigMat.rows != (int)foundCheeseBoardMat.total()){
		std::cout << "foundCheeseBoardMat items not match extrinsic items." << std::endl;
		return false;
	}

	///// load attitude datas
	cv::Mat attitudeBigMat;
	vector<String> xmlFiles;
	cv::glob(folder + "/*.xml", xmlFiles);		 // specific prefix "robotpos##.xml"
	// exits sequence: 1,2,..,10,11,..,20,21
	//		will get 1,10,11,12,...,19,2,20, 21

	// name sort with lambda comparasion
	auto getSubstr = [](const cv::String& str) {   // nums "##" in "robotpos##.xml"
		size_t strPos = str.find("robotpos") + 8;  
		size_t dotPos = str.find_last_of(".");
		return str.substr(strPos, dotPos - strPos);
	};
	std::sort(xmlFiles.begin(), xmlFiles.end(), [&getSubstr](String& lhs, String& rhs) {
		return std::stoi(getSubstr(lhs))<std::stoi(getSubstr(rhs));
	});

	// check items
	if (xmlFiles.size() == 0) {
		std::cout << folder <<  "has no xml files." << std::endl;	return false;	}
	else if (xmlFiles.size() < (int)foundCheeseBoardMat.total())	{
		std::cout << folder << "xml files items do not match found cheese'es count." << std::endl;	return false;	}

	// implementation of loading extrinsic parameters according to foundCheeseBoard
	for (int i = 0; i < foundCheeseBoardMat.total(); ++i){	
		if (foundCheeseBoardMat.at<int>(0,i) == 1)	{
			cv::FileStorage fs(xmlFiles[i], cv::FileStorage::READ);
			if (!fs.isOpened()) {
				std::cout << "Can not open \"" << file << "\"."<< std::endl;	return false;
			}
			vector<double> tmp_10; // specific items 10
			for (int i = 0; i < 10; ++i) { 
				double val = fs[format("Position%d", i)].real();
				tmp_10.push_back(val);
			}				
			fs.release();
			attitudeBigMat.push_back(Mat(tmp_10).t());
		}
	}

	// vector -=> matrix
	for (int i = 0; i < extBigMat.rows; ++i) {
		Mat tmp = attitudeVectorToMatrix(extBigMat.row(i), false); // native cv Rodrigues
		vecHc.push_back(std::move(tmp)); // extrinsic matrix
	}
	for (int i = 0; i < attitudeBigMat.rows; ++i) {
		Mat tmp = attitudeVectorToMatrix(attitudeBigMat.row(i), true);  // use quaternion
		vecHg.push_back(std::move(tmp)); // attitude matrix
	}

	return true;
}

cv::Mat HandEyeCalibration::attitudeVectorToMatrix(cv::Mat& m, 
	bool useQuaternion , const std::string& seq)
{
	// 6  {x,y,z, rx,ry,rz}
	// 10 {x,y,z, qw,qx,qy,qz, rx,ry,rz}

	CV_Assert(m.total() == 6 || m.total() == 10);
	if (m.cols == 1)
		m = m.t();

	Mat tmp = Mat::eye(4, 4, CV_64FC1);

	if (useQuaternion)	// normalized vector, its norm should be 1.
	{
		Vec4d quaternionVec = m({ 3, 0, 4, 1 });
		quaternionToRotatedMatrix(quaternionVec).copyTo(tmp({ 0, 0, 3, 3 }));
		// cout << norm(quaternionVec) << endl; 
	}
	else {
		Mat rotVec;
		if (m.total() == 6) 			
			rotVec = m({ 3, 0, 3, 1 });		//6
		else 	
			rotVec = m({ 7, 0, 3, 1 });		//10

		if (0 == seq.compare("")) 		
			cv::Rodrigues(rotVec, tmp({ 0, 0, 3, 3 }));
		else
			eulerAngleToRotatedMatrix(rotVec, seq).copyTo(tmp({ 0, 0, 3, 3 }));
	}
	tmp({ 3, 0, 1, 3 }) = m({ 0, 0, 3, 1 }).t();
	//std::swap(m,tmp);
	return tmp;
}

cv::Mat HandEyeCalibration::eulerAngleToRotatedMatrix(const cv::Mat& eulerAngle, const std::string& seq)
{
	CV_Assert(eulerAngle.rows == 1 && eulerAngle.cols == 3);

	eulerAngle /= 180 / CV_PI;
	Matx13d m(eulerAngle);
	auto rx = m(0, 0), ry = m(0, 1), rz = m(0, 2);
	auto xs = std::sin(rx), xc = std::cos(rx);
	auto ys = std::sin(ry), yc = std::cos(ry);
	auto zs = std::sin(rz), zc = std::cos(rz);

	Mat rotX = (Mat_<double>(3, 3) << 1, 0, 0, 0, xc, -xs, 0, xs, xc);
	Mat rotY = (Mat_<double>(3, 3) << yc, 0, ys, 0, 1, 0, -ys, 0, yc);
	Mat rotZ = (Mat_<double>(3, 3) << zc, -zs, 0, zs, zc, 0, 0, 0, 1);

	cv::Mat rotMat;
	
	if (seq == "zyx")		rotMat = rotX*rotY*rotZ;
	else if (seq == "yzx")	rotMat = rotX*rotZ*rotY;
	else if (seq == "zxy")	rotMat = rotY*rotX*rotZ;
	else if (seq == "xzy")	rotMat = rotY*rotZ*rotX;
	else if (seq == "yxz")	rotMat = rotZ*rotX*rotY;
	else if (seq == "xyz")	rotMat = rotZ*rotY*rotX;
	else {
		cv::error(cv::Error::StsAssert, "Euler angle sequence string is wrong.",
			__FUNCTION__,__FILE__,__LINE__);
	}

	if (!isRotationMatrix(rotMat))	{
		cv::error(cv::Error::StsAssert, "Euler angle can not convert to rotated matrix",
			__FUNCTION__, __FILE__, __LINE__);
	}

	return rotMat;
	//cout << isRotationMatrix(rotMat) << endl;
}


/** @brief 四元数转旋转矩阵
*
*	@note  数据类型double； 四元数定义 q = w + x*i + y*j + z*k
*
*	@param q 四元数输入{w,x,y,z}向量
*
*	@return 返回旋转矩阵3*3
*/
cv::Mat HandEyeCalibration::quaternionToRotatedMatrix(const cv::Vec4d& q)
{
	double w = q[0], x = q[1], y = q[2], z = q[3];

	double x2 = x * x, y2 = y * y, z2 = z * z;
	double xy = x * y, xz = x * z, yz = y * z;
	double wx = w * x, wy = w * y, wz = w * z;

	Matx33d res{
		1 - 2 * (y2 + z2),	2 * (xy - wz),		2 * (xz + wy),
		2 * (xy + wz),		1 - 2 * (x2 + z2),	2 * (yz - wx),
		2 * (xz - wy),		2 * (yz + wx),		1 - 2* (x2 + y2),
	};

	return Mat(res);
}


// orthogonal matrix... 
//			A = inv(A) = A.t,  A*A.t = A*inv(A) = A.t*A = inv(A)*A = I
bool HandEyeCalibration::isRotationMatrix(const cv::Mat & R)
{
	cv::Mat tmp33 = R({0,0,3,3});
	cv::Mat shouldBeIdentity;

	shouldBeIdentity = tmp33.t()*tmp33;

	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

	return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

/*
*  DELETED
*
// 图像坐标（给定高度），手眼矩阵，机械手臂末端姿态矩阵，外参，内参
cv::Mat HandEyeCalibration::solveLocation(cv::Vec3d& pos, cv::Mat& Hcg, 
	cv::Mat& Hg, cv::Mat& extParamMatrix, cv::Mat& camMatrix)
{
	cv::Mat camMatrixExtend;
	cv::hconcat(camMatrix, cv::Mat::zeros(3, 1, CV_64F), camMatrixExtend);

	//Mat M = camMatrixExtend*Hcg*Hg.inv();   // size 3*4

	Mat M = camMatrixExtend*extParamMatrix*Hcg*Hg;

	// Mat M = camMatrixExtend*extParamMatrix*Hcg;

	//      { u }
	//     z { v } = M * {x,y,z,1}'
	//      { 1 }
	//

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
*/

/** @brief 根据图像坐标(校正)求解在基坐标系下的位置
*
*	@note  数据类型double；返回值（参考矩阵，若使用）的单位与姿态矩阵单位一致
*
*	@param 	imgPt				图像坐标 (u,v)
*	@param 	z					高度
*	@param 	Hcg					手眼标定变换矩阵 4*4
*	@param 	Hg					末端姿态矩阵 4*4
*	@param 	camMatrix			相机内参矩阵 3*3
*	@param 	refHg				参考姿态矩阵（保留）
*	@param 	refExtMatrix		参考外参矩阵（保留）
*
*	@return 空间三维点坐标
*/
cv::Point3d HandEyeCalibration::getWorldPos(cv::Point2d& imgPt, double z, 
	cv::Mat& Hcg, cv::Mat& Hg, cv::Mat& camMatrix, 
	cv::Mat& refHg, cv::Mat& refExtMatrix)
{
	CV_Assert(Hcg.type() == CV_64FC1 && Hg.type() == CV_64FC1 && camMatrix.type() == CV_64FC1);

#if 0	
	double  u = imgPt.x, v = imgPt.y;

	cv::Mat M = camMatrix*Mat::eye(3, 4, CV_64F)*Hcg.inv()*Hg;
	//cv::Mat M = camMatrix*Mat::eye(3, 4, CV_64F)*Hcg*Hg;

	cv::Matx34d MM(M);

	cv::Mat M1 = (Mat_<double>(2, 2) <<
		u*MM(2, 0) - MM(0, 0), u*MM(2, 1) - MM(0, 1),
		v*MM(2, 0) - MM(1, 0), v*MM(2, 1) - MM(1, 1));
	cv::Mat Mr = (Mat_<double>(2, 1) <<
		MM(0, 3) - u*MM(2, 3) - z*(u*MM(2, 2) - MM(0, 2)),
		MM(1, 3) - v*MM(2, 3) - z*(v*MM(2, 2) - MM(1, 2)));

	cv::Mat res(3, 1, CV_64F, cv::Scalar(z));
	res({ 0,0,1,2 }) = M1.inv()*Mr;
#endif

	if (!refHg.empty() && !refExtMatrix.empty())	
	{
		Mat routeRef = refHg*Hcg.inv()*refExtMatrix;
	//	Mat routePredPri = refHg*Hcg.inv();
		Mat routePredPriInv = Hcg*refHg.inv();
	//	Mat predExtMat = routePredPri.inv()*routeRef;
		Mat predExtMat = routePredPriInv*routeRef;
		z = predExtMat.at<double>(2, 3);
	}

	Mat imgPosHomo = Mat::ones(3, 1, CV_64F);
	Mat(imgPt).copyTo(imgPosHomo({ 0,0,1,2 }));
	
	Mat camPosHomo = Mat::ones(4, 1, CV_64F);
	Mat camMatInv = camMatrix.inv(cv::DECOMP_SVD);
	Mat camPos = camMatInv*z*imgPosHomo;
	camPos.copyTo(camPosHomo({ 0,0,1,3 }));

	Mat resHomo = Hg * Hcg.inv()*camPosHomo;
	Mat res = resHomo({ 0,0,1,3 });

	return cv::Point3d(res);
}


/** DELETE

std::vector<cv::Point3d> HandEyeCalibration::getWorldPos(
	std::vector<cv::Point2d>& imgPts, double z,
	cv::Mat& Hcg, cv::Mat& Hg, cv::Mat& camMatrix,
	cv::Mat& refHg, cv::Mat& refExtMatrix)
{
	CV_Assert(Hcg.type() == CV_64FC1 && Hg.type() == CV_64FC1 && camMatrix.type() == CV_64FC1);

	//cv::Mat M = camMatrix*Mat::eye(3, 4, CV_64F)*Hcg.inv()*Hg;
	cv::Mat M = camMatrix*Mat::eye(3, 4, CV_64F)*Hcg*Hg;
	cv::Matx34d MM(M);

	std::vector<cv::Point3d> resPts;
	for (auto& imgPt : imgPts)
	{
		double  u = imgPt.x, v = imgPt.y;
		cv::Mat M1 = (Mat_<double>(2, 2) <<
			u*MM(2, 0) - MM(0, 0), u*MM(2, 1) - MM(0, 1),
			v*MM(2, 0) - MM(1, 0), v*MM(2, 1) - MM(1, 1));
		cv::Mat Mr = (Mat_<double>(2, 1) <<
			MM(0, 3) - u*MM(2, 3) - z*(u*MM(2, 2) - MM(0, 2)),
			MM(1, 3) - v*MM(2, 3) - z*(v*MM(2, 2) - MM(1, 2)));

		cv::Mat res(3, 1, CV_64F, cv::Scalar(z));
		res({ 0,0,1,2 }) = M1.inv()*Mr;

		resPts.emplace_back(std::move(cv::Point3d(res)));
	}
	return resPts;
}
*/

bool HandEyeCalibration::readCameraParameters(const std::string& filename,
	cv::Mat & camMatrix, cv::Mat & distCoefs)
{
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) {
		std::cout << "open \"" << filename << "\" failed." << std::endl;
		return false;
	}
	camMatrix = fs["camera_matrix"].mat();
	distCoefs = fs["distortion_coefficients"].mat();
	fs.release();

	return true;
}

