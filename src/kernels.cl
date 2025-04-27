// Author: Christopher Savini

int2 clip(int2 coordinate, int width, int height)
{
	coordinate.x = max(0, coordinate.x);
	coordinate.y = max(0, coordinate.y);
	coordinate.x = min(width - 1, coordinate.x);
	coordinate.y = min(height - 1, coordinate.y);
	return coordinate;
}

int CoordinateToIndex(int X, int Y, int dim)
{
	return (dim * Y) + X;
}

__kernel void AdaptiveBackgroundSubtraction
(
	__global float * img,
	__global float * background,
	__global float * signal,
	int Width,
	int Height,
	float average,
	__global float * P,
	__global float * AVG,
	int tau
)
{
	int nIndex = 0;
	int neighbourhoodSize = tau;
	int neighbourhoodArea = ((neighbourhoodSize * 2 + 1)*(neighbourhoodSize * 2 + 1));

	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int2 nh;
	int n = CoordinateToIndex(x, y, Width);

	signal[n] = 0;
	for (int i = -neighbourhoodSize; i <= neighbourhoodSize; i++)
	{
		for (int j = -neighbourhoodSize; j <= neighbourhoodSize; j++)
		{
			nh = (int2)(x + i, y + j);
			nh = clip(nh, Width, Height);
			nIndex = CoordinateToIndex(nh.x, nh.y, Width);

			signal[n] = signal[n] + (img[n] - (P[nIndex] * background[n]) + average - AVG[nIndex]);
		}
	}
	signal[n] = signal[n] / neighbourhoodArea;

}

__kernel void CalcPAVG
(
	__global float  * img,
	__global float  * background,
	__global float  * dIdx,
	__global float  * dIdy,
	__global float  * dBdx,
	__global float  * dBdy,
	int Width,
	int Height,
	__global float  * P,
	__global float  * AVG,
	int tau
)
{
	int nIndex = 0;
	int neighbourhoodSize = tau;
	int neighbourhoodArea = ((neighbourhoodSize * 2 + 1)*(neighbourhoodSize * 2 + 1));

	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int2 nh;

	float DBDB = 0;
	float DBDI = 0;

	for (int i = -neighbourhoodSize; i <= neighbourhoodSize; i++)
	{
		for (int j = -neighbourhoodSize; j <= neighbourhoodSize; j++)
		{
			nh = (int2)(x + i, y + j);
			nh = clip(nh, Width, Height);
			nIndex = CoordinateToIndex(nh.x, nh.y, Width);

			DBDB = DBDB + (dBdx[nIndex] * dBdx[nIndex]) + (dBdy[nIndex] * dBdy[nIndex]);
			DBDI = DBDI + (dBdx[nIndex] * dIdx[nIndex]) + (dBdy[nIndex] * dIdy[nIndex]);
		}
	}

	int n = CoordinateToIndex(x, y, Width);
	P[n] = DBDI / DBDB;

	AVG[n] = 0;
	for (int i = -neighbourhoodSize; i <= neighbourhoodSize; i++)
	{
		for (int j = -neighbourhoodSize; j <= neighbourhoodSize; j++)
		{
			nh = (int2)(x + i, y + j);
			nh = clip(nh, Width, Height);
			nIndex = CoordinateToIndex(nh.x, nh.y, Width);

			AVG[n] = AVG[n] + (img[nIndex] - (P[n] * background[nIndex]));
		}
	}

	AVG[n] = AVG[n] / neighbourhoodArea;
}



__kernel void CalcDBDI
(
	
	__global float * img,
	__global float * background,
	__global float * dIdx,
	__global float * dIdy,
	__global float * dBdx,
	__global float * dBdy, 
	int Width,
	int Height
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	int2 coord = (int2)(x, y);
	int2 coordN = (int2)(x, y - 1);
	int2 coordS = (int2)(x, y + 1);
	int2 coordE = (int2)(x + 1, y);
	int2 coordW = (int2)(x - 1, y);
	int2 coordNN = (int2)(x, y - 2);
	int2 coordSS = (int2)(x, y + 2);
	int2 coordEE = (int2)(x + 2, y);
	int2 coordWW = (int2)(x - 2, y);
	int2 coordNNN = (int2)(x, y - 3);
	int2 coordSSS = (int2)(x, y + 3);
	int2 coordEEE = (int2)(x + 3, y);
	int2 coordWWW = (int2)(x - 3, y);
	int2 coordNNNN = (int2)(x, y - 4);
	int2 coordSSSS = (int2)(x, y + 4);
	int2 coordEEEE = (int2)(x + 4, y);
	int2 coordWWWW = (int2)(x - 4, y);
	int2 coordNNNNN = (int2)(x, y - 5);
	int2 coordSSSSS = (int2)(x, y + 5);
	int2 coordEEEEE = (int2)(x + 5, y);
	int2 coordWWWWW = (int2)(x - 5, y);
	int2 coordNNNNNN = (int2)(x, y - 6);
	int2 coordSSSSSS = (int2)(x, y + 6);
	int2 coordEEEEEE = (int2)(x + 6, y);
	int2 coordWWWWWW = (int2)(x - 6, y);
	int2 coordNNNNNNN = (int2)(x, y - 7);
	int2 coordSSSSSSS = (int2)(x, y + 7);
	int2 coordEEEEEEE = (int2)(x + 7, y);
	int2 coordWWWWWWW = (int2)(x - 7, y);
	int2 coordNNNNNNNN = (int2)(x, y - 8);
	int2 coordSSSSSSSS = (int2)(x, y + 8);
	int2 coordEEEEEEEE = (int2)(x + 8, y);
	int2 coordWWWWWWWW = (int2)(x - 8, y);

	coordN = clip(coordN, Width, Height);
	coordS = clip(coordS, Width, Height);
	coordE = clip(coordE, Width, Height);
	coordW = clip(coordW, Width, Height);
	coordNN = clip(coordN, Width, Height);
	coordSS = clip(coordS, Width, Height);
	coordEE = clip(coordE, Width, Height);
	coordWW = clip(coordW, Width, Height);
	coordNNN = clip(coordNNN, Width, Height);
	coordSSS = clip(coordSSS, Width, Height);
	coordEEE = clip(coordEEE, Width, Height);
	coordWWW = clip(coordWWW, Width, Height);
	coordNNNN = clip(coordNNNN, Width, Height);
	coordSSSS = clip(coordSSSS, Width, Height);
	coordEEEE = clip(coordEEEE, Width, Height);
	coordWWWW = clip(coordWWWW, Width, Height);
	coordNNNNN = clip(coordNNNNN, Width, Height);
	coordSSSSS = clip(coordSSSSS, Width, Height);
	coordEEEEE = clip(coordEEEEE, Width, Height);
	coordWWWWW = clip(coordWWWWW, Width, Height);
	coordNNNNNN = clip(coordNNNNNN, Width, Height);
	coordSSSSSS = clip(coordSSSSSS, Width, Height);
	coordEEEEEE = clip(coordEEEEEE, Width, Height);
	coordWWWWWW = clip(coordWWWWWW, Width, Height);
	coordNNNNNNN = clip(coordNNNNNNN, Width, Height);
	coordSSSSSSS = clip(coordSSSSSSS, Width, Height);
	coordEEEEEEE = clip(coordEEEEEEE, Width, Height);
	coordWWWWWWW = clip(coordWWWWWWW, Width, Height);
	coordNNNNNNNN = clip(coordNNNNNNNN, Width, Height);
	coordSSSSSSSS = clip(coordSSSSSSSS, Width, Height);
	coordEEEEEEEE = clip(coordEEEEEEEE, Width, Height);
	coordWWWWWWWW = clip(coordWWWWWWWW, Width, Height);

	int index = CoordinateToIndex(coord.x, coord.y, Width);
	int indexN = CoordinateToIndex(coordN.x, coordN.y, Width);
	int indexS = CoordinateToIndex(coordS.x, coordS.y, Width);
	int indexE = CoordinateToIndex(coordE.x, coordE.y, Width);
	int indexW = CoordinateToIndex(coordW.x, coordW.y, Width);
	int indexNN = CoordinateToIndex(coordNN.x, coordNN.y, Width);
	int indexSS = CoordinateToIndex(coordSS.x, coordSS.y, Width);
	int indexEE = CoordinateToIndex(coordEE.x, coordEE.y, Width);
	int indexWW = CoordinateToIndex(coordWW.x, coordWW.y, Width);
	int indexNNN = CoordinateToIndex(coordNNN.x, coordNNN.y, Width);
	int indexSSS = CoordinateToIndex(coordSSS.x, coordSSS.y, Width);
	int indexEEE = CoordinateToIndex(coordEEE.x, coordEEE.y, Width);
	int indexWWW = CoordinateToIndex(coordWWW.x, coordWWW.y, Width);
	int indexNNNN = CoordinateToIndex(coordNNNN.x, coordNNNN.y, Width);
	int indexSSSS = CoordinateToIndex(coordSSSS.x, coordSSSS.y, Width);
	int indexEEEE = CoordinateToIndex(coordEEEE.x, coordEEEE.y, Width);
	int indexWWWW = CoordinateToIndex(coordWWWW.x, coordWWWW.y, Width);
	int indexNNNNN = CoordinateToIndex(coordNNNNN.x, coordNNNNN.y, Width);
	int indexSSSSS = CoordinateToIndex(coordSSSSS.x, coordSSSSS.y, Width);
	int indexEEEEE = CoordinateToIndex(coordEEEEE.x, coordEEEEE.y, Width);
	int indexWWWWW = CoordinateToIndex(coordWWWWW.x, coordWWWWW.y, Width);
	int indexNNNNNN = CoordinateToIndex(coordNNNNNN.x, coordNNNNNN.y, Width);
	int indexSSSSSS = CoordinateToIndex(coordSSSSSS.x, coordSSSSSS.y, Width);
	int indexEEEEEE = CoordinateToIndex(coordEEEEEE.x, coordEEEEEE.y, Width);
	int indexWWWWWW = CoordinateToIndex(coordWWWWWW.x, coordWWWWWW.y, Width);
	int indexNNNNNNN = CoordinateToIndex(coordNNNNNNN.x, coordNNNNNNN.y, Width);
	int indexSSSSSSS = CoordinateToIndex(coordSSSSSSS.x, coordSSSSSSS.y, Width);
	int indexEEEEEEE = CoordinateToIndex(coordEEEEEEE.x, coordEEEEEEE.y, Width);
	int indexWWWWWWW = CoordinateToIndex(coordWWWWWWW.x, coordWWWWWWW.y, Width);
	int indexNNNNNNNN = CoordinateToIndex(coordNNNNNNNN.x, coordNNNNNNNN.y, Width);
	int indexSSSSSSSS = CoordinateToIndex(coordSSSSSSSS.x, coordSSSSSSSS.y, Width);
	int indexEEEEEEEE = CoordinateToIndex(coordEEEEEEEE.x, coordEEEEEEEE.y, Width);
	int indexWWWWWWWW = CoordinateToIndex(coordWWWWWWWW.x, coordWWWWWWWW.y, Width);

	float I = img[index];
	float IN = img[indexN];
	float IS = img[indexS];
	float IE = img[indexE];
	float IW = img[indexW];
	float INN = img[indexNN];
	float ISS = img[indexSS];
	float IEE = img[indexEE];
	float IWW = img[indexWW];
	float INNN = img[indexNNN];
	float ISSS = img[indexSSS];
	float IEEE = img[indexEEE];
	float IWWW = img[indexWWW];
	float INNNN = img[indexNNNN];
	float ISSSS = img[indexSSSS];
	float IEEEE = img[indexEEEE];
	float IWWWW = img[indexWWWW];
	float INNNNN = img[indexNNNNN];
	float ISSSSS = img[indexSSSSS];
	float IEEEEE = img[indexEEEEE];
	float IWWWWW = img[indexWWWWW];
	float INNNNNN = img[indexNNNNNN];
	float ISSSSSS = img[indexSSSSSS];
	float IEEEEEE = img[indexEEEEEE];
	float IWWWWWW = img[indexWWWWWW];
	float INNNNNNN = img[indexNNNNNNN];
	float ISSSSSSS = img[indexSSSSSSS];
	float IEEEEEEE = img[indexEEEEEEE];
	float IWWWWWWW = img[indexWWWWWWW];
	float INNNNNNNN = img[indexNNNNNNNN];
	float ISSSSSSSS = img[indexSSSSSSSS];
	float IEEEEEEEE = img[indexEEEEEEEE];
	float IWWWWWWWW = img[indexWWWWWWWW];

	dIdx[index] = ((7 * IEEEEEEEE) - (128 * IEEEEEEE) + (1120 * IEEEEEE) - (6272 * IEEEEE) + (25480 * IEEEE) - (81536 * IEEE) + (224224 * IEE) - (640640 * IE) + (640640 * IW) - (224224 * IWW) + (81536 * IWWW) - (25480 * IWWWW) + (6272 * IWWWWW) - (1120 * IWWWWWW) + (128 * IWWWWWWW) - (7 * IWWWWWWWW)) / 720720;
	dIdy[index] = ((7 * ISSSSSSSS) - (128 * ISSSSSSS) + (1120 * ISSSSSS) - (6272 * ISSSSS) + (25480 * ISSSS) - (81536 * ISSS) + (224224 * ISS) - (640640 * IS) + (640640 * IN) - (224224 * INN) + (81536 * INNN) - (25480 * INNNN) + (6272 * INNNNN) - (1120 * INNNNNN) + (128 * INNNNNNN) - (7 * INNNNNNNN)) / 720720;

	float B = background[index];
	float BN = background[indexN];
	float BS = background[indexS];
	float BE = background[indexE];
	float BW = background[indexW];
	float BNN = background[indexNN];
	float BSS = background[indexSS];
	float BEE = background[indexEE];
	float BWW = background[indexWW];
	float BNNN = background[indexNNN];
	float BSSS = background[indexSSS];
	float BEEE = background[indexEEE];
	float BWWW = background[indexWWW];
	float BNNNN = background[indexNNNN];
	float BSSSS = background[indexSSSS];
	float BEEEE = background[indexEEEE];
	float BWWWW = background[indexWWWW];
	float BNNNNN = background[indexNNNNN];
	float BSSSSS = background[indexSSSSS];
	float BEEEEE = background[indexEEEEE];
	float BWWWWW = background[indexWWWWW];
	float BNNNNNN = background[indexNNNNNN];
	float BSSSSSS = background[indexSSSSSS];
	float BEEEEEE = background[indexEEEEEE];
	float BWWWWWW = background[indexWWWWWW];
	float BNNNNNNN = background[indexNNNNNNN];
	float BSSSSSSS = background[indexSSSSSSS];
	float BEEEEEEE = background[indexEEEEEEE];
	float BWWWWWWW = background[indexWWWWWWW];
	float BNNNNNNNN = background[indexNNNNNNNN];
	float BSSSSSSSS = background[indexSSSSSSSS];
	float BEEEEEEEE = background[indexEEEEEEEE];
	float BWWWWWWWW = background[indexWWWWWWWW];

	dBdx[index] = ((7 * BEEEEEEEE) - (128 * BEEEEEEE) + (1120 * BEEEEEE) - (6272 * BEEEEE) + (25480 * BEEEE) - (81536 * BEEE) + (224224 * BEE) - (640640 * BE) + (640640 * BW) - (224224 * BWW) + (81536 * BWWW) - (25480 * BWWWW) + (6272 * BWWWWW) - (1120 * BWWWWWW) + (128 * BWWWWWWW) - (7 * BWWWWWWWW)) / 720720;
	dBdy[index] = ((7 * BSSSSSSSS) - (128 * BSSSSSSS) + (1120 * BSSSSSS) - (6272 * BSSSSS) + (25480 * BSSSS) - (81536 * BSSS) + (224224 * BSS) - (640640 * BS) + (640640 * BN) - (224224 * BNN) + (81536 * BNNN) - (25480 * BNNNN) + (6272 * BNNNNN) - (1120 * BNNNNNN) + (128 * BNNNNNNN) - (7 * BNNNNNNNN)) / 720720;
}


__kernel void CalcDF
(
	__global float * img,
	__global float * dIdx,
	__global float * dIdy,
	int Width,
	int Height
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	int2 coord = (int2)(x, y);
	int2 coordN = (int2)(x, y - 1);
	int2 coordS = (int2)(x, y + 1);
	int2 coordE = (int2)(x + 1, y);
	int2 coordW = (int2)(x - 1, y);
	int2 coordNN = (int2)(x, y - 2);
	int2 coordSS = (int2)(x, y + 2);
	int2 coordEE = (int2)(x + 2, y);
	int2 coordWW = (int2)(x - 2, y);
	int2 coordNNN = (int2)(x, y - 3);
	int2 coordSSS = (int2)(x, y + 3);
	int2 coordEEE = (int2)(x + 3, y);
	int2 coordWWW = (int2)(x - 3, y);
	int2 coordNNNN = (int2)(x, y - 4);
	int2 coordSSSS = (int2)(x, y + 4);
	int2 coordEEEE = (int2)(x + 4, y);
	int2 coordWWWW = (int2)(x - 4, y);
	int2 coordNNNNN = (int2)(x, y - 5);
	int2 coordSSSSS = (int2)(x, y + 5);
	int2 coordEEEEE = (int2)(x + 5, y);
	int2 coordWWWWW = (int2)(x - 5, y);
	int2 coordNNNNNN = (int2)(x, y - 6);
	int2 coordSSSSSS = (int2)(x, y + 6);
	int2 coordEEEEEE = (int2)(x + 6, y);
	int2 coordWWWWWW = (int2)(x - 6, y);
	int2 coordNNNNNNN = (int2)(x, y - 7);
	int2 coordSSSSSSS = (int2)(x, y + 7);
	int2 coordEEEEEEE = (int2)(x + 7, y);
	int2 coordWWWWWWW = (int2)(x - 7, y);
	int2 coordNNNNNNNN = (int2)(x, y - 8);
	int2 coordSSSSSSSS = (int2)(x, y + 8);
	int2 coordEEEEEEEE = (int2)(x + 8, y);
	int2 coordWWWWWWWW = (int2)(x - 8, y);

	coordN = clip(coordN, Width, Height);
	coordS = clip(coordS, Width, Height);
	coordE = clip(coordE, Width, Height);
	coordW = clip(coordW, Width, Height);
	coordNN = clip(coordN, Width, Height);
	coordSS = clip(coordS, Width, Height);
	coordEE = clip(coordE, Width, Height);
	coordWW = clip(coordW, Width, Height);
	coordNNN = clip(coordNNN, Width, Height);
	coordSSS = clip(coordSSS, Width, Height);
	coordEEE = clip(coordEEE, Width, Height);
	coordWWW = clip(coordWWW, Width, Height);
	coordNNNN = clip(coordNNNN, Width, Height);
	coordSSSS = clip(coordSSSS, Width, Height);
	coordEEEE = clip(coordEEEE, Width, Height);
	coordWWWW = clip(coordWWWW, Width, Height);
	coordNNNNN = clip(coordNNNNN, Width, Height);
	coordSSSSS = clip(coordSSSSS, Width, Height);
	coordEEEEE = clip(coordEEEEE, Width, Height);
	coordWWWWW = clip(coordWWWWW, Width, Height);
	coordNNNNNN = clip(coordNNNNNN, Width, Height);
	coordSSSSSS = clip(coordSSSSSS, Width, Height);
	coordEEEEEE = clip(coordEEEEEE, Width, Height);
	coordWWWWWW = clip(coordWWWWWW, Width, Height);
	coordNNNNNNN = clip(coordNNNNNNN, Width, Height);
	coordSSSSSSS = clip(coordSSSSSSS, Width, Height);
	coordEEEEEEE = clip(coordEEEEEEE, Width, Height);
	coordWWWWWWW = clip(coordWWWWWWW, Width, Height);
	coordNNNNNNNN = clip(coordNNNNNNNN, Width, Height);
	coordSSSSSSSS = clip(coordSSSSSSSS, Width, Height);
	coordEEEEEEEE = clip(coordEEEEEEEE, Width, Height);
	coordWWWWWWWW = clip(coordWWWWWWWW, Width, Height);

	int index = CoordinateToIndex(coord.x, coord.y, Width);
	int indexN = CoordinateToIndex(coordN.x, coordN.y, Width);
	int indexS = CoordinateToIndex(coordS.x, coordS.y, Width);
	int indexE = CoordinateToIndex(coordE.x, coordE.y, Width);
	int indexW = CoordinateToIndex(coordW.x, coordW.y, Width);
	int indexNN = CoordinateToIndex(coordNN.x, coordNN.y, Width);
	int indexSS = CoordinateToIndex(coordSS.x, coordSS.y, Width);
	int indexEE = CoordinateToIndex(coordEE.x, coordEE.y, Width);
	int indexWW = CoordinateToIndex(coordWW.x, coordWW.y, Width);
	int indexNNN = CoordinateToIndex(coordNNN.x, coordNNN.y, Width);
	int indexSSS = CoordinateToIndex(coordSSS.x, coordSSS.y, Width);
	int indexEEE = CoordinateToIndex(coordEEE.x, coordEEE.y, Width);
	int indexWWW = CoordinateToIndex(coordWWW.x, coordWWW.y, Width);
	int indexNNNN = CoordinateToIndex(coordNNNN.x, coordNNNN.y, Width);
	int indexSSSS = CoordinateToIndex(coordSSSS.x, coordSSSS.y, Width);
	int indexEEEE = CoordinateToIndex(coordEEEE.x, coordEEEE.y, Width);
	int indexWWWW = CoordinateToIndex(coordWWWW.x, coordWWWW.y, Width);
	int indexNNNNN = CoordinateToIndex(coordNNNNN.x, coordNNNNN.y, Width);
	int indexSSSSS = CoordinateToIndex(coordSSSSS.x, coordSSSSS.y, Width);
	int indexEEEEE = CoordinateToIndex(coordEEEEE.x, coordEEEEE.y, Width);
	int indexWWWWW = CoordinateToIndex(coordWWWWW.x, coordWWWWW.y, Width);
	int indexNNNNNN = CoordinateToIndex(coordNNNNNN.x, coordNNNNNN.y, Width);
	int indexSSSSSS = CoordinateToIndex(coordSSSSSS.x, coordSSSSSS.y, Width);
	int indexEEEEEE = CoordinateToIndex(coordEEEEEE.x, coordEEEEEE.y, Width);
	int indexWWWWWW = CoordinateToIndex(coordWWWWWW.x, coordWWWWWW.y, Width);
	int indexNNNNNNN = CoordinateToIndex(coordNNNNNNN.x, coordNNNNNNN.y, Width);
	int indexSSSSSSS = CoordinateToIndex(coordSSSSSSS.x, coordSSSSSSS.y, Width);
	int indexEEEEEEE = CoordinateToIndex(coordEEEEEEE.x, coordEEEEEEE.y, Width);
	int indexWWWWWWW = CoordinateToIndex(coordWWWWWWW.x, coordWWWWWWW.y, Width);
	int indexNNNNNNNN = CoordinateToIndex(coordNNNNNNNN.x, coordNNNNNNNN.y, Width);
	int indexSSSSSSSS = CoordinateToIndex(coordSSSSSSSS.x, coordSSSSSSSS.y, Width);
	int indexEEEEEEEE = CoordinateToIndex(coordEEEEEEEE.x, coordEEEEEEEE.y, Width);
	int indexWWWWWWWW = CoordinateToIndex(coordWWWWWWWW.x, coordWWWWWWWW.y, Width);

	float I = img[index];
	float IN = img[indexN];
	float IS = img[indexS];
	float IE = img[indexE];
	float IW = img[indexW];
	float INN = img[indexNN];
	float ISS = img[indexSS];
	float IEE = img[indexEE];
	float IWW = img[indexWW];
	float INNN = img[indexNNN];
	float ISSS = img[indexSSS];
	float IEEE = img[indexEEE];
	float IWWW = img[indexWWW];
	float INNNN = img[indexNNNN];
	float ISSSS = img[indexSSSS];
	float IEEEE = img[indexEEEE];
	float IWWWW = img[indexWWWW];
	float INNNNN = img[indexNNNNN];
	float ISSSSS = img[indexSSSSS];
	float IEEEEE = img[indexEEEEE];
	float IWWWWW = img[indexWWWWW];
	float INNNNNN = img[indexNNNNNN];
	float ISSSSSS = img[indexSSSSSS];
	float IEEEEEE = img[indexEEEEEE];
	float IWWWWWW = img[indexWWWWWW];
	float INNNNNNN = img[indexNNNNNNN];
	float ISSSSSSS = img[indexSSSSSSS];
	float IEEEEEEE = img[indexEEEEEEE];
	float IWWWWWWW = img[indexWWWWWWW];
	float INNNNNNNN = img[indexNNNNNNNN];
	float ISSSSSSSS = img[indexSSSSSSSS];
	float IEEEEEEEE = img[indexEEEEEEEE];
	float IWWWWWWWW = img[indexWWWWWWWW];

	dIdx[index] = ((7 * IEEEEEEEE) - (128 * IEEEEEEE) + (1120 * IEEEEEE) - (6272 * IEEEEE) + (25480 * IEEEE) - (81536 * IEEE) + (224224 * IEE) - (640640 * IE) + (640640 * IW) - (224224 * IWW) + (81536 * IWWW) - (25480 * IWWWW) + (6272 * IWWWWW) - (1120 * IWWWWWW) + (128 * IWWWWWWW) - (7 * IWWWWWWWW)) / 720720;
	dIdy[index] = ((7 * ISSSSSSSS) - (128 * ISSSSSSS) + (1120 * ISSSSSS) - (6272 * ISSSSS) + (25480 * ISSSS) - (81536 * ISSS) + (224224 * ISS) - (640640 * IS) + (640640 * IN) - (224224 * INN) + (81536 * INNN) - (25480 * INNNN) + (6272 * INNNNN) - (1120 * INNNNNN) + (128 * INNNNNNN) - (7 * INNNNNNNN)) / 720720;

}

