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
        std::cout << "Set InitPositions" << std::endl;
        scatterParticles(&box->p, box->L);
        cudaMemcpy(box->p.diam, box->p.diam_dev, NP * sizeof(float),cudaMemcpyDeviceToHost);
        for(uint n = 0; n < NP; n++){
            box->positionFile << box->p.diam[n] << " ";
            box->animeFile << box->p.diam[n] << " ";
        }
        box->positionFile << std::endl << std::endl;
        box->animeFile << std::endl << std::endl;
        //set posMem and list
        setIntVecZero<<<NB,NT>>>(box->g.cell_dev, box->g.M * box->g.M * box->g.EpM);
        updateGrid2D<<<NB,NT>>>(box->g, box->g.cell_dev, box->p.x_dev);
        //remove overraps by using harmonic potential
        uint Nt = 20. / box->dt;
        for(int nt = 0; nt < Nt; nt++){
            harmonicEvoBox(box);
        }
        
        std::cout << "-> SIP Done!" << std::endl;
        return;
    }
    void initBox(Box* box, uint ID){
        box->id = ID;
        std::cout << "Start Initialisation: ID = " << box->id << std::endl;
    
        //for record
        std::ostringstream positionFileName;
        positionFileName << "../pos/N" << (uint)NP << "/T" << Tfin << "/posBD_N" << (uint)NP << "_T" << Tfin << "_id" << box->id <<".data";
        box->positionFile.open(positionFileName.str().c_str());
        std::ostringstream animeFileName;
        animeFileName << "../anime/N" << (uint)NP << "/T" << Tfin << "/animeBD_N" << (uint)NP << "_T" << Tfin << "_id" << box->id <<".log";
        box->animeFile.open(animeFileName.str().c_str());
        
        setdt_T(box, dt_INIT, Tfin);
        prepareBox(box);
        setdt_T(box, dt_BD, Tfin);
        equilibrateBox(box, tmax);
        std::cout << "-> Init Done!" << std::endl;
        return;
    }

    //time evolution
    inline void harmonicEvoBox(Box* box){
        culcHarmonicFint2D<<<NB,NT>>>(
            box->g, 
            box->g.refCell_dev, 
            box->g.cell_dev, 
            box->p.force_dev, 
            box->L, 
            box->p.diam_dev, 
            box->p.x_dev
        );
        vEvoBD<<<NB,NT>>>(box->p.v_dev, box->dt, 0, box->p.force_dev, box->p.rndState_dev);
        removevg2D(&box->p);
        xEvo<<<NB,NT>>>(box->p.x_dev, box->dt, box->L, box->p.v_dev);
        checkUpdate(&box->g, box->dt, box->p.x_dev, box->p.v_dev);

        return;
    }
    inline void tEvoBox(Box* box){
        culcFint2D<<<NB,NT>>>(
            box->g, 
            box->g.refCell_dev, 
            box->g.cell_dev, 
            box->p.force_dev, 
            box->L, 
            box->p.diam_dev, 
            box->p.x_dev
        );
        vEvoBD<<<NB,NT>>>(box->p.v_dev, box->dt, box->thermalFuctor, box->p.force_dev, box->p.rndState_dev);
        removevg2D(&box->p);
        xEvo<<<NB,NT>>>(box->p.x_dev, box->dt, box->L, box->p.v_dev);
        checkUpdate(&box->g, box->dt, box->p.x_dev, box->p.v_dev);
        
        return;
    }

    //equilibrations
    void equilibrateBox(Box* box, double teq){
        std::cout << "Equilibrate the System: ID = " << box->id << std::endl;
	    uint Nt = teq/box->dt;
	    for (uint nt = 0; nt < Nt; nt++) {
		    tEvoBox(box);
	    }
	    std::cout << " -> Edone"<< box->id << std::endl;
        return;
    }

    //record
    void recBox(std::ofstream *of, Box* box){
        cudaMemcpy(box->p.x, box->p.x_dev, D * NP * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(box->p.v, box->p.v_dev, D * NP * sizeof(float), cudaMemcpyDeviceToHost);
        *of << box->t << " " << K(&box->p) << " " << U(&box->g, box->p.diam_dev, box->p.x_dev) << " ";
	    for (int n = 0; n < NP; n++) {
		    for (char d = 0; d < D; d++) {
			    *of << box->p.x_dev[d*NP+n] << " ";
            }
            for (char d = 0; d < D; d++) {
			    *of << box->p.v_dev[d*NP+n] << " ";
            }
        }
	    *of << std::endl;
        return;
    }
    void getData(Box* box){
        std::cout << "Starting time loop: ID = " << box->id << std::endl;
        uint Nt, tag;
        std::cout << "getting anime datas in 10secs" << std::endl;
        Nt = 10./box->dt;
        tag = 0;
        for(uint nt = 0; nt <=Nt; nt++){
            tEvoBox(box);
            if(nt >= tag){
                box->t = nt * box->dt;
                recBox(&box->animeFile, box);
                tag += tag = 0.1/box->dt;
            }
        }
        std::cout << "getting logPlot datas" << std::endl;
        Nt = tmax/box->dt;
        tag = 10;
        for(uint nt = 0; nt <= Nt; nt++){
            tEvoBox(box);
            if(nt >= tag){
                box->t = nt * box->dt;
                recBox(&box->positionFile, box);
                tag *= 1.1;
            }
        }
        return;
    }
}