#include "../h/box.cuh"

namespace PhysPeach{
    //setters and getters
    void makeBox(Box* box){

        makeParticles(&box->p);

        //default settings
        box->id = 0;
        box->dt = dt_MD;
        box->T = Tfin;
        box->L = sqrt((double)NP/(double)DNSTY);
        box->thermalFuctor = sqrt(2*box->T/box->dt);
        
        box->LDDir = "/LD";
        box->MDDir = "/MD";
        box->EDir = "/E";
        box->posDir = "/pos";
        box->velDir = "/vel";

        //Make dir tree
        std::ostringstream trajName;
        struct stat st;
        trajName << "../traj/N" << (uint)NP;
        if(stat(trajName.str().c_str(), &st) != 0){
            mkdir(trajName.str().c_str(), 0755);
            std::cout << "created " << trajName.str() << std::endl;
        }
        trajName << "/T" << Tfin;
        box->NTDir = trajName.str();
        if(stat(box->NTDir.c_str(), &st) != 0){
            mkdir(box->NTDir.c_str(), 0755);
            std::cout << "created " << box->NTDir << std::endl;

            mkdir((box->NTDir + box->LDDir).c_str(), 0755);
            mkdir((box->NTDir + box->LDDir + box->EDir).c_str(), 0755);
            mkdir((box->NTDir + box->LDDir + box->posDir).c_str(), 0755);
            mkdir((box->NTDir + box->LDDir + box->velDir).c_str(), 0755);
            
            mkdir((box->NTDir + box->MDDir).c_str(), 0755);
            mkdir((box->NTDir + box->MDDir + box->EDir).c_str(), 0755);
            mkdir((box->NTDir + box->MDDir + box->posDir).c_str(), 0755);
            mkdir((box->NTDir + box->MDDir + box->velDir).c_str(), 0755);
        }

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
        struct stat st;
        std::ostringstream diamName;
        diamName << "../traj/N" << NP << "/diam.data";
        if(stat(diamName.str().c_str(), &st) != 0){
            std::ofstream diamFile;
            diamFile.open(diamName.str().c_str());
            std::cout << "created " << diamName.str() << std::endl;
            cudaMemcpy(box->p.diam, box->p.diam_dev, NP * sizeof(float),cudaMemcpyDeviceToHost);
            for(uint n = 0; n < NP; n++){
                diamFile << box->p.diam[n] << std::endl;
            }
            diamFile.close();
        }
        //set posMem and list
        setIntVecZero<<<NB,NT>>>(box->g.cell_dev, box->g.M * box->g.M * box->g.EpM);
        updateGrid2D<<<NB,NT>>>(box->g, box->g.cell_dev, box->p.x_dev);
        //remove overraps by using harmonic potential
        uint Nt = 10. / box->dt;
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
        culcFint2D<<<IB,IT>>>(
            box->g, 
            box->g.refCell_dev, 
            box->g.cell_dev, 
            box->p.force_dev, 
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
    void recPos(std::ofstream *of, Box* box){
        cudaMemcpy(box->p.x, box->p.x_dev, D * NP * sizeof(double), cudaMemcpyDeviceToHost);
	    for (int n = 0; n < NP; n++) {
		    for (char d = 0; d < D; d++) {
			    *of << box->p.x[d*NP+n] << " ";
            }
        }
        *of << std::endl;
        return;
    }
    void getData(Box* box){
        std::cout << "Starting time loop: ID = " << box->id << std::endl;
        uint Nt, tag;
        std::ofstream tFile;
        std::ofstream eFile;
        std::ofstream posFile;

        if(box->id == 1){
            std::cout << "getting liniarPlot datas in 5 secs" << std::endl;

            std::string tLinpltName = "/tliniar.data";
            tFile.open((box->NTDir + box->LDDir + tLinpltName).c_str());

            std::ostringstream eLinpltName;
            eLinpltName << box->NTDir + box->LDDir + box->EDir << "/liniar.data";
            eFile.open(eLinpltName.str().c_str());

            std::ostringstream posLinpltName;
            posLinpltName << box->NTDir + box->LDDir + box->posDir << "/liniar.data";
            posFile.open(posLinpltName.str().c_str());

            Nt = 5./box->dt;
            tag = 0;
            for(uint nt = 0; nt < Nt; nt++){
                tEvoBox(box);
                if(nt >= tag){
                    if(box->id == 1){
                        tFile << nt * box->dt << std::endl;
                    }
                    eFile << K(&box->p) << " " << U(&box->g, box->p.diam_dev, box->p.x_dev) << std::endl;
                    recPos(&posFile, box);
                    tag += 0.1/box->dt;
                }
            }
            posFile.close();
            eFile.close();
            tFile.close();
            std::string tLogpltName = "/tlog.data";
            tFile.open((box->NTDir + box->LDDir + tLogpltName).c_str());
        }

        std::cout << "getting logPlot datas" << std::endl;
        std::ostringstream eLogpltName;
        eLogpltName << box->NTDir + box->LDDir + box->EDir << "/id" << box->id << ".data";
        eFile.open(eLogpltName.str().c_str());

        std::ostringstream posLogpltName;
        posLogpltName << box->NTDir + box->LDDir + box->posDir << "/id" << box->id << ".data";
        posFile.open(posLogpltName.str().c_str());

        Nt = tmax/box->dt;
        tag = 10;
        for(uint nt = 0; nt <= Nt; nt++){
            tEvoBox(box);
            if(nt >= tag){
                if(box->id == 1){
                    tFile << nt * box->dt << std::endl;
                }
                eFile << K(&box->p) << " " << U(&box->g, box->p.diam_dev, box->p.x_dev) << std::endl;
                recPos(&posFile, box);
                tag *= 1.3;
            }
        }
        if(box->id == 1){
            tFile.close();
        }
        eFile.close();
        posFile.close();
        std::cout << "Every steps have been done: ID = " << box->id << std::endl << std::endl;
        return;
    }
    void benchmark(Box* box, uint loop){
        for(uint l = 0; l <=loop; l++){
            tEvoBox(box);
        }
        return;
    }
}