#ifndef PARAMETERS_CUH
#define PARAMETERS_CUH
//--Define device parameters--
//Numbers of Threads and Blocks for general use
#define NT 512
#define NB 8
//Numbers of Threads and Blocks for Interactions
#define NG 2
////IT = NG * NG * EpM: IB = ((unsigned int)0.9+M/NG)^2: M:=(int)sqrt(N/dnsty)/4.3a0
extern unsigned int IT;
#define IB 9

//--Define scale parameters--
//Num of particles
#define N 1000
//Numbers of initial condition
extern unsigned int IDs;
extern unsigned int IDe;

//--Define parameters--
#define PI 3.141593

//time constant
#define dt_INIT 0.01
#define dt_BD 0.003
#define dt_MD 0.001
extern float tau;

//dimentions
#define D 2

//particles diameters
#define a0 1.2 //(a1+a2)/2
#define a1 1.0
#define a2 1.4

//density of particles(V = N / Density)
#define DNSTY 0.8

//friction parameter: (ZT/m)sqrt(m*a^2/ep)
//ZT = 1.0;

//temparature parameters
#define Tinit 4.
extern float Tfin;

#endif