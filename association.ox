#include<oxstd.h>
#include<oxprob.h>
#include<oxfloat.h>

main()
{										
	
	decl time = timer();
    decl nsim, th, nburn, K, R, L, mD, vyt, mst, mit, vi, ct, P, amLam, l, vphi0;
	decl vphi1, mz, mmu, meta, msamp, mh0, kk, vd, vpi, mL, j, mh, vsig2, vmu;
	decl mSig, i, vh, vh0, vf, mf1, vy, ms, mH, mH0, mH1, mP, vi1, vl;
	decl amHyx, vHy, mHx, mG, vP, vrho2;

	/*--- Seed ---*/											

	ranseed(14); 	   // set a positive integer	

	/*--- set sampling option ---*/

	nsim = 500;			    // # of collected MCMC iteration
	th = 10;				// # of skimming, nsim x th = total # of MCMC iteration
	nburn = 500;			// burn-in period	
	K = 5;					// # of factors
	R = 200;                // # of Monte Carlo simulation for evaluation of \pi(x|y)
	L = 34;                 // # of causes in the data
	
	/*--- load data ---*/
	
	mD = loadmat("data.csv");	

	vyt = mD[][0];		  
	mst = mD[][1:];	 
	mit = (mst .== 999);		
	ct = rows(vyt);
	P = columns(mst);

	/*--- Initial values ---*/
	
	amLam = new array[L];	 
	for(l=0 ; l<L ; l++)
	  amLam[l] = zeros(K, P); 

	vphi0 = vphi1 = ones(1, P); 
	mz = ( mst .> 0 ) - ( mst .<= 0 );
	mmu = zeros(L, P);
	meta = rann(ct, K);
	msamp = <>;
	mh0 = ( vyt .== range(0, L-1) );

	/*--- MCMC sampling ---*/
	
	println("\n\nIteration:");

	/*----------- S A M P L I N G   S T A R T -----------*/
		
	for(kk=-nburn ; kk<th*nsim ; kk++){  // MCMC iteration

    /*--- sampling pi_y ---*/

	vd = randirichlet(1, 1 + sumc( vyt .== range(0, L-1) ));
	vpi = vd ~ (1 - sumr(vd)); 

    /*--- sampling mu ---*/
	
	mL = zeros(ct, P);
	for(l=0 ; l<L ; l++){
      vi = vecindex(vyt .== l);
	  if(rows(vi) > 0)
		mL[vi][] = mz[vi][] - meta[vi][] * amLam[l];
	}// l

	for(j=0 ; j<P ; j++){
	  mh = mh0 .* !mit[][j];
	  vsig2 = 1 ./ (sumc(mh)' + vphi0[j]);
	  vmu = vsig2 .* sumc(mL[][j] .* mh)';
	  mmu[][j] = vmu + sqrt(vsig2) .* rann(L, 1);
	}// j

    /*--- sampling lambda ---*/

	for(l=0 ; l<L ; l++){
	  for(j=0 ; j<P ; j++){
		vi = vecindex( (vyt .== l) .* !mit[][j]);
		if(rows(vi) > 0){
		  mSig = invert(vphi1[j]*unit(K) + meta[vi][]' meta[vi][]);
		  vmu = mSig * meta[vi][]' * (mz[vi][j] - mmu[l][j]);
		  amLam[l][][j] = vmu + choleski(mSig) * rann(K, 1);
		}// if
		else
		  amLam[l][][j] = sqrt(1 ./ vphi1[j]) .* rann(K, 1);
	  }// j
	}// l

    /*--- sampling eta ---*/

	for(i=0 ; i<ct ; i++){
	  vi = vecindex(!mit[i][]);	l = vyt[i];
	  mSig = invert( unit(K) + amLam[l][][vi] * amLam[l][][vi]' );
	  vmu = mSig * amLam[l][][vi] * (mz[i][vi] - mmu[l][vi])';
	  meta[i][] = ( vmu + choleski(mSig) * rann(K, 1) )';
	}// i  

	/*--- sampling tau ---*/
	
	vh = sumc(mmu.^2);
	vphi0 = rangamma(1, P, (1 + L)/2, (1 + vh)/2 );			 	

	/*--- sampling phi ---*/

	vh = zeros(1, P);
	for(l=0 ; l<L ; l++)
	  vh += sumc(amLam[l].^2);
	
	vphi1 = rangamma(1, P, (1 + K*L)/2, (1 + vh)/2 );			 	

	/*--- sampling z ---*/	
	
	for(l=0 ; l<L ; l++){
	  for(j=0 ; j<P ; j++){
	    vi = vecindex((vyt .== l) .* !mit[][j] .* !mst[][j]);
        if(rows(vi) > 0){
		  vmu = mmu[l][j] + meta[vi][] * amLam[l][][j];
		  vh0 = probn( zeros(rows(vi), 1) - vmu );	
		  mz[vi][j] = vmu + quann(vh0 .* ranu(rows(vi), 1));
		  vf = vecindex((mz[vi][j] .== +.Inf) + (mz[vi][j] .== -.Inf));
		  if(rows(vf) > 0)
		    mz[vi[vf]][j] = -1 .* (mst[vi[vf]][j] .== 0) + (mst[vi[vf]][j] .== 1);
	    }// if s == 0
	    vi = vecindex((vyt .== l) .* !mit[][j] .* mst[][j]);
        if(rows(vi) > 0){
		  vmu = mmu[l][j] + meta[vi][] * amLam[l][][j];
		  vh0 = probn( zeros(rows(vi), 1) - vmu );	
		  mz[vi][j] = vmu + quann((1 - vh0) .* ranu(rows(vi), 1) + vh0);
		  vf = vecindex((mz[vi][j] .== +.Inf) + (mz[vi][j] .== -.Inf));
		  if(rows(vf) > 0)
		    mz[vi[vf]][j] = -1 .* (mst[vi[vf]][j] .== 0) + (mst[vi[vf]][j] .== 1);
	    }// if s == 1
	  }// j
	}// l

	/*--- storing sample ---*/

	if(kk >= 0 && !imod(kk, th)){										 
	  
	  /*--- compute delta ---*/

	  amHyx = new array[L];	  
	  mHx = zeros(P, 2);
	  for(l=0 ; l<L ; l++){
		mG = amLam[l]' amLam[l] + unit(P);
		vh = sqrt( diagonal(mG) );
		vP = probn( ( 0 - mmu[l][] ) ./ vh )';
		vP += ( (vP .== 0) - ((1 - vP) .== 0) ) .* 10^(-10);
		amHyx[l] = vpi[l] .* ( vP ~ (1 - vP) );
		mHx += amHyx[l];
	  }// l
	  
	  vrho2 = zeros(1, P);
	  for(j=0 ; j<P ; j++){
		for(l=0 ; l<L ; l++)
		  vrho2[j] += ( amHyx[l][j][0] - vpi[l] * mHx[j][0] )^2 / ( vpi[l] * mHx[j][0] )
				    + ( amHyx[l][j][1] - vpi[l] * mHx[j][1] )^2 / ( vpi[l] * mHx[j][1] );
	  }// j

	  msamp |= sqrt(vrho2);

	}// saving samples
	
	/*--- print counter ---*/

	if(!imod(kk, 100)){println(kk);}	  
										  
	}//k [MCMC]		 	

	/*------------ S A M P L I N G   E N D --------------*/

	/*--- output ---*/
	
	savemat("result-association.csv", msamp);

	println("\n\nTime: ", timespan(time));
}