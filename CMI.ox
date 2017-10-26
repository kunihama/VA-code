#include<oxstd.h>
#include<oxprob.h>
#include<oxfloat.h>
#include<oxdraw.h>

main()
{										
	
	decl time = timer();
    decl nsim, th, nburn, K, R, L, mD, vyt, mst, mit, vi, ct, P, amLam, l, vphi0;
	decl vphi1, mz, mmu, meta, msamp, mh0, kk, vd, vpi, mL, j, mh, vsig2, vmu;
	decl mSig, i, vh, vh0, vf, mf1, vy, ms, mH, mH0, mH1, mP, vi1, vl;

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
	vi = vecindex(meanc(mit) .< 0.05);
	mst = mst[][vi];
	ct = rows(vyt);
	mit = (mst .== 999);		
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

	  /*--- impute missing values ---*/

	  mf1 = zeros(ct, P);
	  for(l=0 ; l<L ; l++){
		vi = vecindex(mh0[][l]);
		if(rows(vi) > 0){
		  mh = mmu[l][] + meta[vi][] * amLam[l] + rann(rows(vi), P);
		  mf1[vi][] = mh .> 0;
		}// if
      }// l
	  vy = vyt;
	  ms = !mit .* mst + mit .* mf1;
	  
	  /*--- compute cmi ---*/

	  mH = zeros(ct, 2);
	  mH0 = mH1 = zeros(ct, P);
	  for(l=0 ; l<L ; l++){
	    mL = zeros(ct, R);
	    mP = probn( - (mmu[l][] + rann(R, K) * amLam[l]) );
		mP += ( (log(mP) .== -.Inf) - (log(1-mP) .== -.Inf) ) .* 10^(-10);

	    for(j=0 ; j<P ; j++)
		  mL += !ms[][j] .* log(mP[][j])' + (ms[][j] .== 1) .* log(1 - mP[][j])';

		if(any(mL .== -.Inf))
		  println("something wrong in log calculation!");

		vh = meanr(exp(mL));
		vi1 = vecindex(vy .== l);
		mH[vi1][0] = vh[vi1] .* vpi[l];
	    mH[][1] += vh .* vpi[l];

	    for(j=0 ; j<P ; j++){
		  mh = mL - (!ms[][j] .* log(mP[][j])' + (ms[][j] .== 1) .* log(1 - mP[][j])');		
		  vh = meanr(exp(mh));
		  mH0[vi1][j] = vh[vi1] * vpi[l];
		  mH1[][j] += vh * vpi[l];
		}// j
	  }// l

	  vl = zeros(1, P);
	  mh = log(mH[][0]) - log(mH0) - log(mH[][1]) + log(mH1);
	  for(j=0 ; j<P ; j++){
		vi = vecindex((mh[][j] .!= .NaN) .* (mh[][j] .!= -.Inf) .* (mh[][j] .!= +.Inf));
		vl[j] = meanc(mh[vi][j]);
	  }// j

	  msamp |= vl;

	}// saving samples
	
	/*--- print counter ---*/

	if(!imod(kk, 100)){println(kk);}	  
										  
	}//k [MCMC]		 	

	/*------------ S A M P L I N G   E N D --------------*/

	/*--- output ---*/

	savemat("result-cmi.csv", msamp);

	println("\n\nTime: ", timespan(time));
}