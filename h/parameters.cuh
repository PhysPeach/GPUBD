#ifndef PARAMETERS_CUH
#define PARAMETERS_CUH

typedef  unsigned int uint;

//--Settings--
//Numbers of Threads and Blocks for general use
const uint NT  = 512;
const uint NB = 4;

//Num of Particles
const uint NP = 1024;
//dimentions , density of box(V = N / Density)
const uint D = 2;
const float DNSTY = 0.8;

//M = sqrt(Num of Cells)
const uint SQRT_NUM_OF_CELLS = 6;
//Elements Num per Cells
const uint EPM = 56;

//Numbers of Threads and Blocks for Interactions
const uint NGx = 3;
const uint NGy = 3;
const uint IT = 512;
const uint IB = 4;

//--Define scale parameters--
//Numbers of initial condition
extern uint IDs;
extern uint IDe;

//--Define parameters--
const float PI = 3.141593;

//time constant
const float dt_INIT = 0.01;
const float dt_BD = 0.003;
const float dt_MD = 0.001;
extern double tmax;

//particles diameters
const float a0 = 1.2; //(a1+a2)/2
const float a1 = 1.0;
const float a2 = 1.4;

//friction parameter: (ZT/m)sqrt(m*a^2/ep)
//ZT = 1.0;

//temparature parameters
const float Tinit = 4.;
extern float Tfin;

#endif