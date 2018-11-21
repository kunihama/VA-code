#include<oxstd.h>
#include<oxprob.h>
#include<oxfloat.h>
#include<oxdraw.h>

main()
{											
	decl time = timer();
	decl K, H, P, Q, L, i, k, p, q, j, l, h, r, R, B, b, d, m;
	decl nsim, nburn, msamp, cn, da, vs, mmu, th, ms, vmu, vsig2, mh; 
	decl mL, vh0, vf, mb, meta, mG, mSig, mM, mP, ct, vy0, vi, vyt, mst, mit;
    decl kk, vpi, amLam, vpi0, vphi0, vphi1, vh, dk, vy, vsig, mz, mD, mys;
	decl md0, md1;
	
	/*--- Seed ---*/											

	ranseed(12); 	   // set a positive integer	

	/*--- set sampling option ---*/

	nsim = 500;			    // # of collected MCMC iteration
	th = 10;				// # of skimming, nsim x th = total # of MCMC iteration
	nburn = 500;			// burn-in period	
	K = 5; 					// # of factors
	R = 200; 				// # of Monte Carlo simulation for evaluation of \pi(x|y)
	L = 34;					// # of causes in the data
	
	/*--- load data ---*/
	
	mD = loadmat("test.csv");	 	// load target data 
	mb = loadmat("training.csv");	// load training data

	vy0 = vy = mD[][0];
	ms = mD[][1:];
	mM = (ms .== 999);				// 999: missing data
	vi = vecindex(meanc(mM) .< 1);
	ms = ms[][vi];
	cn = rows(vy);
	P = columns(ms);
	mM = (ms .== 999);		   

    mD = mb;	
	vyt = mD[][0];
	mst = mD[][1:];
	mst = mst[][vi];
	mit = (mst .== 999); 
	ct  = rows(vyt);
	
	/*--- Initial values ---*/
	
	amLam = new array[L];	 
	for(l=0 ; l<L ; l++)
	  amLam[l] = zeros(K, P); 

	vphi0 = vphi1 = ones(1, P); 
	mz = ( mst .> 0 ) - ( mst .<= 0 );
	mmu = zeros(L, P);
	meta = rann(ct, K);	
	vpi0 = meanc(vy0 .== range(0, L-1));
	vpi  = meanc(vyt .== range(0, L-1));
	msamp = <>;	

    md0 = !mM  .* !ms;  md1 = !mM  .* ms;	

	/*--- MCMC sampling ---*/
	
	println("\n\nIteration:");

	/*----------- S A M P L I N G   S T A R T -----------*/
		
	for(kk=-nburn ; kk<th*nsim ; kk++){  // MCMC iteration

    /*--- sampling mu ---*/
	
	mL = zeros(ct, P);
	for(l=0 ; l<L ; l++){
      vi = vecindex(vyt .== l);
	  if(rows(vi) > 0)
		mL[vi][] = mz[vi][] - meta[vi][] * amLam[l];
	}// l

	for(j=0 ; j<P ; j++){
	  mh = (vyt .== range(0, L-1)) .* !mit[][j];
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

	/*--- estimate distribution in a target site ---*/

	  mh = zeros(cn, L);
	  for(l=0; l<L; l++){
		mL = mmu[l][] + rann(R, K) * amLam[l];
		mP = probn( -mL ); 
		mP += ( (log(mP) .== -.Inf) - (log(1-mP) .== -.Inf) ) .* 10^(-10);
	    mG = md0 * log(mP)' + md1 * log(1 - mP)';	  
		mh[][l] = meanr( exp(mG) );
	  }// l
	  vy = sumr(cumulate( (mh ./ sumr(mh))' )' .< ranu(cn, 1));	

	  vpi = meanc(vy .== range(0, L-1));
	  vh = 1 - sumr(fabs(vpi0 - vpi)) / ( 2 * (1 - min(vpi0)) ); // CSMF	  
	  msamp |= vh ~ vpi;
	}// saving samples
	
	/*--- print counter ---*/

	if(!imod(kk, 100)){println(kk);}	  
										  
	}//k [MCMC]		 	

	/*------------ S A M P L I N G   E N D --------------*/

	/*--- output ---*/

	savemat("result.csv", msamp);

	println("\n\nTime: ", timespan(time));
}
