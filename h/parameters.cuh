#ifndef PARAMETERS_CUH
#define PARAMETERS_CUH

typedef  unsigned int uint;

//--Define device parameters--
//Numbers of Threads and Blocks for general use
const uint NT  = 512;
const uint NB = 8;
//Numbers of Threads and Blocks for Interactions
const uint NG  = 4;
////IT = NG * NG * EpM: IB = (()0.9+M/NG)sqrt(N/dnsty)/4.3a0
const uint IT = 1024;
const uint IB = 3;

//--Define scale parameters--
//Num of particles
const uint NP = 1024;
//Numbers of initial condition
extern uint IDs;
extern uint IDe;

//--Define parameters--
const float PI = 3.141593;

//time constant
const float dt_INIT = 0.01;
const float dt_BD = 0.003;
const float dt_MD = 0.001;
extern float tau;

//dimentions
const uint D = 2;

//particles diameters
const float a0 = 1.2; //(a1+a2)/2
const float a1 = 1.0;
const float a2 = 1.4;

//density of particles(V = N / Density)
const float DNSTY = 0.8;

//friction parameter: (ZT/m)sqrt(m*a^2/ep)
//ZT = 1.0;

//temparature parameters
const float Tinit = 4.;
extern float Tfin;

#endif