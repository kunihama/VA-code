#include<oxstd.h>
#include<oxprob.h>
#include<oxfloat.h>
#include<oxdraw.h>

main()
{											
	decl time = timer();
    decl nsim, th, nburn, L, dp, mD, mb, vy0, vy, ms, mM, vi, cn, vd, mh;												   
	decl P, vyt, mst, mit, ct, amPsi, l, j, J, vpi0, msamp, kk, vc, vpi, vh;
	 
	/*--- Seed ---*/											

	ranseed(12); 	   // set a positive integer	

	/*--- set sampling option ---*/

	nsim = 500;			    // # of collected MCMC iteration
	th = 10;				// # of skimming, nsim x th = total # of MCMC iteration
	nburn = 500;			// burn-in period	
	L = 34;                 // # of cause
	
	/*--- load data ---*/
	
	mD = loadmat("test.csv");	 	// load target data 
	mb = loadmat("training.csv");	// load training data
	  
	vy0 = vy = mD[][0];
	ms = mD[][1:];
	mM = (ms .== 999);		
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
	
	amPsi = new array[L];	 
	for(l=0 ; l<L ; l++){
	  amPsi[l] = new array[P];	 
	  for(j=0 ; j<P ; j++)
	    amPsi[l][j] = ones(1, 2) / 2; 
	}// l
	  
	vpi0 = meanc(vy0 .== range(0, L-1));
	msamp = <>;	

	/*--- MCMC sampling ---*/
	
	println("\n\nIteration:");

	/*----------- S A M P L I N G   S T A R T -----------*/
		
	for(kk=-nburn ; kk<th*nsim ; kk++){  // MCMC iteration
	
	/*--- sampling pi(x_j|y) ---*/

	for(l=0 ; l<L ; l++){
	  for(j=0 ; j<P ; j++){
		vc = zeros(1, 2);
		vi = vecindex((vyt .== l) .* !mit[][j]);
	    if(any(vi))
		  vc = sumc( mst[vi][j] .== range(0, 1) );

		vd = randirichlet(1, 1 + vc);
		amPsi[l][j] = vd ~ (1 - sumr(vd)); 
	  }// j
	}// l

	/*--- storing sample ---*/

	if(kk >= 0 && !imod(kk, th)){										 

	  /*--- estimate distribution in a target site ---*/

	  mh = zeros(cn, L);
	  for(l=0 ; l<L ; l++){
	    for(j=0 ; j<P ; j++)
		  mh[][l] += ( !mM[][j] .* !ms[][j] ) .* log(amPsi[l][j][0]) + ( !mM[][j] .* ms[][j] ) .* log(amPsi[l][j][1]);
	  }// l
	  mh = exp(mh);
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

	savemat("result-ci.csv", msamp);

	println("\n\nTime: ", timespan(time));
}
