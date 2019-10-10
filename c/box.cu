#include "../h/box.cuh"

namespace PhysPeach{
    //setters and getters
    void makeBox(Box* box){

        makeParticles(&box->p);

        //default settings
        box->id = 0;
        box->dt = dt_MD;
        box->t = 0.;
        box->T = Tfin;
        box->L = sqrt(NP/DNSTY);
        box->thermalFuctor = sqrt(2*box->T/box->dt);

        makeGrid(&box->g, box->L);

        std::cout << "Made Box" << std::endl;
        
        return;
    }

    void killBox(Box* box){

        //others
        killGrid(&box->g);
        killParticles(&box->p);
        std::cout << "Killed Box" << std::endl;

        return;
    }

    void prepareBox(Box* box){
        box->logFile << "Set InitPositions" << std::endl;
        scatterParticles(&box->p, box->L);
        cudaMemcpy(box->p.diam, box->p.diam_dev, NP * sizeof(float),cudaMemcpyDeviceToHost);
        for(uint n = 0; n < NP; n++){
            box->positionFile << box->p.diam[n] << " ";
        }
        box->positionFile << std::endl << std::endl;
        //set posMem and list
        setIntVecZero<<<NB,NT>>>(box->g.cell_dev, box->g.M * box->g.M * box->g.EpM);
        updateGrid2D<<<NB,NT>>>(box->g, box->g.cell_dev, box->p.x_dev);
        //remove overraps by using harmonic potential
        uint Nt = 20. / box->dt;
        /*for(int nt = 0; nt < Nt; nt++){
            tHarmonicDvlp();
            judgeUpdateGrid();
        }*/
        
        box->logFile << "-> SIP Done!" << std::endl;
        return;
    }
    void initBox(Box* box, uint ID){
        box->id = ID;
        std::cout << "Start Initialisation: ID = " << box->id << std::endl;
    
       //for record
        std::ostringstream positionFileName;
        positionFileName << "../pos/N" << (uint)NP << "/T" << Tfin << "/posBD_N" << (uint)NP << "_T" << Tfin << "_id" << box->id <<".data";
        box->positionFile.open(positionFileName.str().c_str());
        std::ostringstream logFileName;
        logFileName << "../log/N" << (uint)NP << "/T" << Tfin << "/logBD_N" << (uint)NP << "_T" << Tfin << "_id" << box->id <<".log";
        box->logFile.open(logFileName.str().c_str());

        box->logFile << "Start Initialisation: ID = " << box->id << std::endl;
        box->logFile << "Created Box ID = " << box->id << std::endl;
        
        //hotstart(Tinit >> 1)
        setdt_T(box, dt_INIT, Tinit);

        prepareBox(box);
    
        //equilibrateSys(30.0);
    
        //Tinit -> Tfin
        //coolSys(Tfin, tau);
    
        //setdt_T(dt_BD, Tfin);
    
        //equilibrateSys(10 * tau);
    
        box->logFile << "-> Init Done!" << std::endl;
        return;
    }

    //time evolution
    inline void harmonicEvoBox(Box* box){
        //culcHarmonicInteractions
        //vDvlpBD
        //setvgzero2D
        //xDvlp
        //checkgrid

        return;
    }
    inline void tEvoBox(Box* box){
        //culcHarmonicInteractions
        //vDvlpBD
        //setvgzero2D
        //xDvlp
        //checkgrid
        
        return;
    }
}