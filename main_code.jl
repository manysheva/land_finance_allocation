using Random, Distributions
using Printf
using Plots
using LaTeXStrings
using DataFrames
Plots.PyPlotBackend()
using CSV
using JLD2
pyplot()

"""
Base code for the model in "Land Property Rights, Financial Frictions,
    and Resource Allocation in Developing Countries", Kristina Manysheva
Illustrating solutions to the stationary equilibrium
with value function iterations, and other basic techniques.
"""
"""
Parts of this code benifited a lot from the computational techiques 
covered in Econ 416-2 "Advanced Macroeconomics" at Northwestern University
taught by Matthew Rognlie
"""

function stationary(Pi,pi="None",atol=1E-10,maxit=10000)
    """computes stationary distribution of Markov chain with transition Pi via
    simple iteration until subsequent iterations have difference below atol.
    option to start iteration with selected pi"""
    if pi=="None"
        pi=ones(1,size(Pi)[1])./size(Pi)[1]
    end

    for it=1:maxit
        pi_new = pi * Pi
            if maximum(abs.(pi_new-pi)) < atol
                break
            end
            pi = pi_new
    end
    return(pi)
end


function mar_tauch(rho, sigma, N, m=3)
    """implements Tauchen method to approximate AR(1) with persistence rho and
    normal innovations with standard deviation sigma, with N discrete states,
    within -m and m  times its stationary sd"""

    Pi =  Array{Float64}(undef, N, N)
    sigma_y = sigma*sqrt(1/(1-rho^2))
    s = range(-m*sigma_y,stop=m*sigma_y,length=N)
    ds = s[2]-s[1]
    d=Normal(0, sigma)
    Pi[:,1]=cdf.(d, s[1].-0.9*s.+ds/2)

    Pi[:,N] = 1 .-cdf.(d, s[N].-0.9*s.-ds/2)
    for j = 2:N-1
        Pi[:,j] = (cdf.(d, s[j].-0.9*s.+ds/2)
                 - cdf.(d, s[j].-0.9*s.-ds/2))
    end
    pi = stationary(Pi)
    return(s, pi, Pi)
end


function markov_product(rho,sigma_y,N)
    """helper method that assumes AR(1) process in logs for productivity and
    scales aggregate productivity to 1"""
    sigma = sigma_y*sqrt(1-rho^2)
    s,pi,Pi = mar_tauch(rho,sigma,N)
    y = exp.(s)/sum(pi*exp.(s))
    return (y,pi,Pi)
end


function combine_markov(x_x,pi_x,Pi_x,y_y,pi_y,Pi_y)
    """combines two _independent_ Markov processes into single process"""
    x = kron(x_x,ones(size(y_y)[1]))
    y = kron(ones(size(x_x)[1]),y_y)
    pi = kron(pi_x,pi_y)
    Pi = kron(Pi_x,Pi_y)
    return (x,y,pi,Pi)
end


function con_pr(rl, r, y, lp, a, aprime)
    """Obtain residual consumption of households with private land"""
    con=y+rl*lp+(1+r)*a.-aprime
    con[con.<0].=0.0000001
    return con
end

function con_com(r, y, a, aprime)
    """Obtain residual consumption of households with communal land"""
    con=y+(1+r)*a.-aprime
    con[con.<0].=0.0000001
    return con
end

function utility(c, eta1)
    """Household utility"""
    return c.^(1-eta1)./(1-eta1)
end

function get_rule_assets(a, apol)
    """obtain the rule for assets"""
    a_pol_i = Array{Int64}(undef, size(apol))
    for i=1:size(apol)[1]
        for j=1:size(apol)[2]
            for k=1:size(apol)[3]
                a_pol_i[i,j,k]= findall(x->x==apol[i,j,k], a)[1]
            end
        end
    end


    return (a_pol_i)
end
function forward_iterate_p(D,Pi,a_pol_i, lp)
    """iterates from distribution D of assets to Dnew,
    given assets policy rule by (a_pol_i), and exogenous Markov
    transition matrix Pi for the exogenous state (productivity)
        for households with private land"""
    Dnew = zeros(size(D))
    for i=1:size(a_pol_i)[1]
        for j=1:size(a_pol_i)[2]
            for k=1:size(a_pol_i)[3]
                Dnew[i,j,a_pol_i[i,j,k]]+=D[i,j,k]
            end
        end
    end


    # now use transition matrix to adjust distribution across 's'
    for j=1:size(lp)[1]
        Dnew[:,j,:] = transpose(Pi) * Dnew[:,j,:]
    end

    return Dnew
end

function invdist_p(dlp, a, apolp, Pi, lp, D="None",atol=1E-10,maxit=10000)
    """finds invariant distribution for private part given assets for endogenous
    state a and Markov transition matrix Pic for exogenous state of productivities, possibly
    with starting distribution D as a seed"""
    pi = stationary(Pi) # compute separately exogenous inv dist to start there
    if D=="None"
        D = repeat(transpose(pic),1,size(lp)[1],size(a)[1]).*repeat(transpose(dlp./(size(a)[1])), size(pic)[2], 1, size(a)[1]) # assume equispaced on grid for assets
    end

    a_pol_i = get_rule(a, apolp) # obtain policy rule

    # now iterate until convergence according to atol, only checking every 10 it
    Dnew = Array{Int64}(undef, size(D))
    for it=1:maxit
        Dnew = forward_iterate_p(D,Pi,a_pol_i,lp)

        if (it % 10 == 0)*(maximum(abs.(Dnew-D)) < atol)
            break
        end
        D = Dnew
    end

    return Dnew

end

function inter_land(nu,lc)
    """interpolates and returns pair: array xdi, giving indices xdi and xdi+1
    in 'x' between which each entry of xdi falls; and array xdpi, giving weight
    xdpi on x[xdi] vs. 1-xdpi on x[xdi+1] for the communal land the household would have
    next period given endogenous amount of land transfer"""

    xdi = Array{Int64}(undef, size(lc)[1], 2)
    xdpi = Array{Float64}(undef, size(lc)[1], 2)

    cur_i=1
    for i=1:size(lc)[1]
        if (i<size(lc)[1])*(cur_i<size(lc)[1])
            l_new=lc[i]+nu
            while cur_i<size(lc)[1]
                if (l_new>lc[cur_i])*(l_new<lc[cur_i+1])
                    xdi[i,1]=cur_i
                    xdi[i,2]=cur_i+1
                    xdpi[i,1]=1-(l_new-lc[cur_i])/(lc[cur_i+1]-lc[cur_i])
                    xdpi[i,2]=1-(lc[cur_i+1]-l_new)/(lc[cur_i+1]-lc[cur_i])
                    break
                elseif (l_new>lc[cur_i])*((cur_i+1)==size(lc)[1])
                    xdi[i,1]=cur_i
                    xdi[i,2]=cur_i+1
                    xdpi[i,1]=1-(l_new-lc[cur_i])/(lc[cur_i+1]-lc[cur_i])
                    xdpi[i,2]=1-xdpi[i,1]
                    break
                elseif l_new==lc[cur_i]
                    xdi[i,1]=cur_i
                    xdi[i,2]=cur_i
                    xdpi[i,1]=0.5
                    xdpi[i,2]=0.5
                    break
                else
                    cur_i+=1
                end
            end

        elseif i==size(lc)[1]
            xdi[i,1]=i
            xdi[i,2]=i
            xdpi[i,1]=0.5
            xdpi[i,2]=0.5
        else
            xdi[i,1]=cur_i
            xdi[i,2]=cur_i
            xdpi[i,1]=0.5
            xdpi[i,2]=0.5
        end
    end


    return (xdi,xdpi)
end

function forward_iterate_c(D,Pi,a_pol_i, ocpolc, index_land, prob_land, pireal, piexpr, lc)
    """iterates from distribution D of s*a to Dnew, given interpolated land rule,
    endogenous asset policy rule, and exogenous Markov
    transition matrix Pi for the productivity in the communal part of the economy"""
    Dnew = zeros(size(D))
    for i=1:size(ocpolc)[1]
        for j=1:size(ocpolc)[2]
            for k=1:size(ocpolc)[3]
                if ocpolc[i,j,k]==2
                    Dnew[i,j,a_pol_i[i,j,k]]+=(1-pireal)*D[i,j,k]
                    Dnew[i,index_land[j,1],a_pol_i[i,j,k]]+=pireal*prob_land[j,1]*D[i,j,k]
                    Dnew[i,index_land[j,2],a_pol_i[i,j,k]]+=pireal*prob_land[j,2]*D[i,j,k]
                else
                    Dnew[i,j,a_pol_i[i,j,k]]+=(1-piexpr)*D[i,j,k]
                    Dnew[i,1,a_pol_i[i,j,k]]+=piexpr*D[i,j,k]
                end

            end
        end
    end

    # now use transition matrix to adjust distribution across 's'

    for j=1:size(lc)[1]
        Dnew[:,j,:] = transpose(Pi) * Dnew[:,j,:]
    end

    return Dnew
end

function invdist_c(dlc, a, apolc, ocpolc, nu, Pi, lc, pireal, piexpr, D="None",atol=1E-10,maxit=100)
    """finds invariant distribution for communal part given assets for endogenous
    state a, land allocation and Markov transition matrix Pic for exogenous state of productivities, possibly
    with starting distribution D as a seed"""
    pi = stationary(Pi) # compute separately exogenous inv dist to start there
    if D=="None"
        D = repeat(transpose(pic),1,size(lc)[1],size(a)[1]).*repeat(transpose(dlc./(size(a)[1])), size(pic)[2], 1, size(a)[1]) # assume equispaced on grid for assets
    end

    a_pol_i = get_rule(a, apolc) # obtain asset policy rule
    new_land=inter_land(nu,lc)
    index_land=new_land[1]
    prob_land=new_land[2] # obtain land policy rule

    # now iterate until convergence according to atol, only checking every 10 it
    Dnew = Array{Int64}(undef, size(D))

    for it=1:maxit

        Dnew = forward_iterate_c(D,Pi,a_pol_i, ocpolc, index_land, prob_land, pireal, piexpr, lc)
        if (it % 10 == 0)*(maximum(abs.(Dnew-D)) < atol)
            break
        end
        D = Dnew
    end

    return Dnew
end

function stationary_dist(dlp, dlc, r, rk, rl, w, a, lp, zac, znc, pic, Pic, beta, alphaf, gammaf,
    alphae, gammae, lambdk, eta1, lc, nu, piexpr, pireal, target_comm)
    """Function that finds stationary distribution for both communal and private part of the economy, 
    finds the share of households for each part to match distribution of land and total share of communal land,
    and computes the aggregate values for capital, assets, labor, and land that used to check whether market clearing conditions are satisfied
    """
    get_all_c = value_iteration_com_rent(r, rk, rl,  w, a, lc,  zac, znc, pic, Pic, nu, beta, eta1, piexpr, pireal, alphaf, gammaf, alphae, gammae, lambdk)
    Dc = invdist_c(dlc, a, get_all_c[6], get_all_c[2], nu, Pic, lc, pireal, piexpr)
    get_policy_pr = value_iteration_pr( r, rk, rl, w, a, lp, zac, znc, pic, Pic, beta, alphaf, gammaf, alphae, gammae, lambdk, eta1)
    Dpr = invdist_p(dlp, a, get_policy_pr[7], Pic, lp, "None")
    
    ac = sum(Dc .* get_all_c[6])
    labc = sum(Dc .* get_all_c[7])
    capc = sum(Dc .* get_all_c[8])
    rentinc = sum(Dc .* get_all_c[4])
    rentoutc = 0
    workerc = sum(Dc .* get_all_c[9])
    Dc_land = zeros(lcnum)
        landc=sum(Dc .* get_all_c[10] .* get_all_c[3])
        for i = 1:lcnum
            Dc_land[i] = sum(Dc[:, i, :])
        end
    ap = sum(Dpr .* get_policy_pr[7])
    labp = sum(Dpr .* get_policy_pr[8])
    capp = sum(Dpr .* get_policy_pr[9])
    rentinp = sum(Dpr .* get_policy_pr[4])
    rentoutp = sum(Dpr .* get_policy_pr[5])
    workerp = sum(Dpr .* get_policy_pr[10])
    
    #Find total share of households living under different property rights regimes
    cland = sum(Dc_land .* lc)
    pland = sum(dlp .* lp)
    lambda1 = (cland * (1 - target_comm)) / (cland + (pland - cland) * target_comm)
    
    #Compute aggregate variables to check market clearing

    at = ap * lambda1 + ac * (1 - lambda1)
    capt = capc * (1 - lambda1) + capp * lambda1        
    labt = labc * (1 - lambda1) + labp * lambda1
    workert = workerc * (1 - lambda1) + workerp * lambda1
    rout = rentoutp * lambda1+rentoutc*(1-lambda1)
    rin = rentinc * (1 - lambda1) + rentinp * lambda1
    realocation = sum(Dc[:, 1:lcnum-1, :] .* pireal .* nu .*get_all_c[10][:, 1:lcnum-1, :])
    expropr = sum(piexpr .* (1 .- get_all_c[10]) .* Dc .*repeat(transpose(lc), size(zac)[1], 1, size(a)[1]))
    
    return (at, capt, labt, workert, realocation, expropr, rin, rout)
       
                                                               



function value_iteration_pr(r, rk, rl,  w, a, lp,  zac, znc, pic, Pic, beta, alphaf, gammaf, alphae, gammae,
                       lambdk, eta1, maxit=10000,atol=1E-5)
"""Value function iteration for households in private part of the economy"""

    Vnewp=Array{Float64}(undef, size(zac)[1], size(lp)[1], size(a)[1])
    fill!(Vnewp, 90)
    cpolp = zeros(size(Vnewp))
    apolp = zeros(size(Vnewp))
    ocpolp=zeros(size(Vnewp))
    lpolp=zeros(size(Vnewp))
    l1polp=zeros(size(Vnewp))
    land_rent_out=zeros(size(Vnewp))
    land_rent_in_p=zeros(size(Vnewp))
    yp=zeros(size(Vnewp))
    yap=zeros(size(Vnewp))
    ynp=zeros(size(Vnewp))
    laborp=zeros(size(Vnewp))
    capitalp=zeros(size(Vnewp))
    workerp=zeros(size(Vnewp))
    farmerp=zeros(size(Vnewp))
    creditp=zeros(size(Vnewp))
    collatp=zeros(size(Vnewp))
    Vworker=0
    Vfarmer=0
    Ventr=0
    i=0
    iteration=0
    diff=1
    Voldp=similar(Vnewp)
    fill!(Voldp, 80)
    atol=1E-5
    maxit=1000
    pl=rl*beta/(1-beta)
    while diff>atol
        iteration+=1

        @assert(iteration < maxit, "value function never converged")
        if iteration % 50 == 0
            @printf("iteration %f", iteration)
            @printf("difference %f", diff)
            #print("diff"+diff)
        end

        Voldp=copy(Vnewp)

        for i =1:size(zac)[1]
            for j=1:size(lp)[1]
                for k=1:size(a)[1]

                    EV=transpose(Pic[i,:])*Vnewp[:,j,:]

                    #Choosing between worker, farmer and entepreneur

                    #Find optimal level of assets for worker
                    #Find max amount of assets prime

                    kprime_max=max(w+rl*lp[j]+(1+r)*a[k],0)
                    amax=a[a.<=kprime_max]
                    cwp_i=con_pr(rl, r, w, lp[j], a[k], amax)
                    u_i=utility(cwp_i, eta1)+beta.*EV[1:size(amax)[1]]
                    kprime_worker=argmax(u_i)
                    Vworker=u_i[kprime_worker]



                    #Now solve the problem for farmer
                    #Get the capital and check the constraint
                    capf=(zac[i]*((gammaf/rl)^gammaf)*((rk/alphaf)^(gammaf-1)))^(1/(1-alphaf-gammaf))
                    if capf>lambdk*(a[k]+pl*lp[j])-pl*lp[j]
                        capf=lambdk*(a[k]+pl*lp[j])-pl*lp[j]
                    end

                    land=(rk/rl)*(gammaf/alphaf)*capf
                    yf=zac[i]*(capf^alphaf)*(land^gammaf)-rk*capf-rl*land
                    kprime_max=max(yf+rl*lp[j]+(1+r)*a[k],0)
                    amax=a[a.<=kprime_max]
                    cfp_i=con_pr(rl, r, yf, lp[j], a[k], amax)
                    u_i=utility(cfp_i, eta1)+beta.*EV[1:size(amax)[1]]
                    kprime_farmer=argmax(u_i)
                    Vfarmer=u_i[kprime_farmer]



                    #Now solve the problem of entrepreneur in similar way to farmer

                    cape=(znc[i]*((gammae/w)^gammae)*((rk/alphae)^(gammae-1)))^(1/(1-alphae-gammae))
                    if cape>lambdk*(a[k]+pl*lp[j])-pl*lp[j]
                        cape=lambdk*(a[k]+pl*lp[j])-pl*lp[j]
                    end

                    hire_labor=(rk/w)*(gammae/alphae)*cape
                    ye=znc[i]*(cape^alphae)*(hire_labor^gammae)-rk*cape-w*hire_labor
                    kprime_max=max(ye+rl*lp[j]+(1+r)*a[k],0)
                    amax=a[a.<=kprime_max]
                    cep_i=con_pr(rl, r, ye, lp[j], a[k], amax)
                    u_i=utility(cep_i, eta1)+beta.*EV[1:size(amax)[1]]
                    kprime_entr=argmax(u_i)
                    Ventr=u_i[kprime_entr]


                    l1polp[i,j,k]=land
                    #Choose the occupation 1-worker, 2-farmer, 3-entepreneur

                    if (Vworker>=Vfarmer)*(Vworker>=Ventr)
                        Vnewp[i,j,k]=Vworker
                        cpolp[i,j,k]=cwp_i[kprime_worker]
                        ocpolp[i,j,k]=1
                        lpolp[i,j,k]=0
                        l1polp[i,j,k]=land
                        land_rent_in_p[i,j,k]=0
                        land_rent_out[i,j,k]=lp[j]
                        yp[i,j,k]=0
                        yap[i,j,k]=0
                        ynp[i,j,k]=0
                        apolp[i,j,k]=a[kprime_worker]
                        laborp[i,j,k]=0
                        capitalp[i,j,k]=0
                        workerp[i,j,k]=1
                        farmerp[i,j,k]=0
                        creditp[i,j,k]=0
                        collatp[i,j,k]=0

                    elseif (Vworker<=Vfarmer)*(Vfarmer>=Ventr)
                        Vnewp[i,j,k]=Vfarmer
                        cpolp[i,j,k]=cfp_i[kprime_farmer]
                        ocpolp[i,j,k]=2
                        lpolp[i,j,k]=land
                        l1polp[i,j,k]=land
                        if land>lp[j]
                            land_rent_in_p[i,j,k]=land-lp[j]
                            land_rent_out[i,j,k]=0
                        else
                            land_rent_in_p[i,j,k]=0
                            land_rent_out[i,j,k]=-land+lp[j]
                        end
                        yp[i,j,k]=zac[i]*(capf^alphaf)*(land^gammaf)
                        yap[i,j,k]=zac[i]*(capf^alphaf)*(land^gammaf)
                        ynp[i,j,k]=0
                        apolp[i,j,k]=a[kprime_farmer]
                        laborp[i,j,k]=0
                        capitalp[i,j,k]=capf
                        workerp[i,j,k]=0
                        farmerp[i,j,k]=1
                        if capf>a[k]
                            creditp[i,j,k]=capf-a[k]
                            collatp[i,j,k]=(1/(lambdk-1))*creditp[i,j,k]
                        else
                            creditp[i,j,k]=0
                            collatp[i,j,k]=0
                        end

                    else
                        Vnewp[i,j,k]=Ventr
                        cpolp[i,j,k]=cep_i[kprime_entr]
                        ocpolp[i,j,k]=3
                        lpolp[i,j,k]=0
                        l1polp[i,j,k]=land
                        land_rent_in_p[i,j,k]=0
                        land_rent_out[i,j,k]=lp[j]
                        yp[i,j,k]=znc[i]*(cape^alphae)*(hire_labor^gammae)
                        yap[i,j,k]=0
                        ynp[i,j,k]=znc[i]*(cape^alphae)*(hire_labor^gammae)
                        apolp[i,j,k]=a[kprime_entr]
                        laborp[i,j,k]=hire_labor
                        capitalp[i,j,k]=cape
                        workerp[i,j,k]=0
                        farmerp[i,j,k]=0
                        if cape>a[k]
                            creditp[i,j,k]=cape-a[k]
                            collatp[i,j,k]=(1/(lambdk-1))*creditp[i,j,k]
                        else
                            creditp[i,j,k]=0
                            collatp[i,j,k]=0
                        end

                    end
                end
            end
        end

        diff=maximum(abs.(Voldp-Vnewp))

    end
    return (cpolp,  ocpolp, lpolp, land_rent_in_p, land_rent_out,
                 yp, apolp, laborp, capitalp, workerp, farmerp, yap,
                 ynp, l1polp, Vnewp, creditp, collatp)
end




function value_iteration_com_rent(r, rk, rl,  w, a, lc,  zac, znc, pic, Pic, nu, beta, eta1, piexpr,
                                                   pireal, alphaf, gammaf, alphae, gammae,
                                    lambdk, maxit=10000,atol=1E-5)
 """Value function iteration for households in communal part of the economy"""

    Vnewc=Array{Float64}(undef, size(zac)[1], size(lc)[1], size(a)[1])
    fill!(Vnewc,90)
    cpolc = zeros(size(Vnewc))
    apolc = zeros(size(Vnewc))
    ocpolc=zeros(size(Vnewc))
    lpolc=zeros(size(Vnewc))
    l1polc=zeros(size(Vnewc))
    land_rent_in_c=zeros(size(Vnewc))
    yc=zeros(size(Vnewc))
    yac=zeros(size(Vnewc))
    ync=zeros(size(Vnewc))
    laborc=zeros(size(Vnewc))
    capitalc=zeros(size(Vnewc))
    workerc=zeros(size(Vnewc))
    farmerc=zeros(size(Vnewc))
    creditc=zeros(size(Vnewc))
    collatc=zeros(size(Vnewc))
    Vworker=0
    Vfarmer=0
    Ventr=0
    i=0
    iteration=0
    diff=1
    Voldc=similar(Vnewc)
    fill!(Voldc,80)
    atol=1E-5
    maxit=1000
    #given nu determine where the person can be reallocated
    new_land=inter_land(nu,lc)
    index_land=new_land[1]
    prob_land=new_land[2]

    while diff>atol
        iteration+=1

        @assert(iteration < maxit, "value function never converged")
        if iteration % 50 == 0
            @printf("iteration %f", iteration)
            @printf("difference %f", diff)
        end

        Voldc=copy(Vnewc)



        for i =1:size(zac)[1]
            for j=1:size(lc)[1]
                for k=1:size(a)[1]

                    EV_sameland=transpose(Pic[i,:])*Vnewc[:,j,:]
                    EV_noland=transpose(Pic[i,:])*Vnewc[:,1,:]
                    EV_moreland1=transpose(Pic[i,:])*Vnewc[:,index_land[j,1],:]
                    EV_moreland2=transpose(Pic[i,:])*Vnewc[:,index_land[j,2],:]

                    #Choosing between worker, farmer and entepreneur

                    #Find optimal level of assets for worker
                    #Find max amount of assets prime

                    kprime_max=max(w+(1+r)*a[k],0)
                    amax=a[a.<=kprime_max]
                    cwp_i=con_com(r, w, a[k], amax)

                    u_i=utility(cwp_i, eta1)+beta.*(piexpr.*EV_noland[1:size(amax)[1]]+(1-piexpr).*EV_sameland[1:size(amax)[1]])
                    kprime_worker=argmax(u_i)
                    Vworker=u_i[kprime_worker]

                    #Now solve the problem for farmer
                    #Get the capital and check the constraint
                    capf=(zac[i]*((gammaf/rl)^gammaf)*((rk/alphaf)^(gammaf-1)))^(1/(1-alphaf-gammaf))
                    if capf>lambdk*a[k]
                        capf=lambdk*a[k]
                    end

                    land=(rk/rl)*(gammaf/alphaf)*capf
                    if land>lc[j]

                        yf=zac[i]*(capf^alphaf)*(land^gammaf)-rk*capf-rl*(land-lc[j])
                    else
                        land=lc[j]

                        capf=(zac[i]*land*alphaf/rk)^(1/(1-alphaf))

                        if capf>lambdk*a[k]
                            capf=lambdk*a[k]
                        end
                        yf=zac[i]*(capf^alphaf)*(land^gammaf)-rk*capf
                    end
                    kprime_max=max(yf+(1+r)*a[k],0)
                    amax=a[a.<=kprime_max]
                    cfp_i=con_com(r, yf, a[k], amax)
                    u_i=utility(cfp_i, eta1)+beta.*((1-pireal).*EV_sameland[1:size(amax)[1]]+pireal.*(prob_land[j,1].*EV_moreland1[1:size(amax)[1]]+prob_land[j,2].*EV_moreland2[1:size(amax)[1]]))
                    kprime_farmer=argmax(u_i)
                    Vfarmer=u_i[kprime_farmer]


                    #Now solve the problem of entrepreneur in similar way to farmer

                    cape=(znc[i]*((gammae/w)^gammae)*((rk/alphae)^(gammae-1)))^(1/(1-alphae-gammae))
                    if cape>lambdk*a[k]
                        cape=lambdk*a[k]
                    end

                    hire_labor=(rk/w)*(gammae/alphae)*cape
                    ye=znc[i]*(cape^alphae)*(hire_labor^gammae)-rk*cape-w*hire_labor
                    kprime_max=max(ye+(1+r)*a[k],0)
                    amax=a[a.<=kprime_max]
                    cep_i=con_com(r, ye, a[k], amax)
                    u_i=utility(cep_i, eta1)+beta.*(piexpr.*EV_noland[1:size(amax)[1]]+(1-piexpr).*EV_sameland[1:size(amax)[1]])
                    kprime_entr=argmax(u_i)
                    Ventr=u_i[kprime_entr]

                    l1polc[i,j,k]=land

                     #Choose the occupation 1-worker, 2-farmer, 3-entepreneur

                    if (Vworker>=Vfarmer)*(Vworker>=Ventr)
                        Vnewc[i,j,k]=Vworker
                        cpolc[i,j,k]=cwp_i[kprime_worker]
                        ocpolc[i,j,k]=1
                        lpolc[i,j,k]=0
                        l1polc[i,j,k]=land
                        land_rent_in_c[i,j,k]=0
                        yc[i,j,k]=0
                        yac[i,j,k]=0
                        ync[i,j,k]=0
                        apolc[i,j,k]=a[kprime_worker]
                        laborc[i,j,k]=0
                        capitalc[i,j,k]=0
                        workerc[i,j,k]=1
                        farmerc[i,j,k]=0
                        creditc[i,j,k]=0
                        collatc[i,j,k]=0

                    elseif (Vworker<=Vfarmer)*(Vfarmer>=Ventr)
                        Vnewc[i,j,k]=Vfarmer
                        cpolc[i,j,k]=cfp_i[kprime_farmer]
                        ocpolc[i,j,k]=2
                        lpolc[i,j,k]=land
                        l1polc[i,j,k]=land
                        if land>lc[j]
                            land_rent_in_c[i,j,k]=land-lc[j]
                        else
                            land_rent_in_c[i,j,k]=0
                        end
                        yc[i,j,k]=zac[i]*(capf^alphaf)*(land^gammaf)
                        yac[i,j,k]=zac[i]*(capf^alphaf)*(land^gammaf)
                        ync[i,j,k]=0
                        apolc[i,j,k]=a[kprime_farmer]
                        laborc[i,j,k]=0
                        capitalc[i,j,k]=capf
                        workerc[i,j,k]=0
                        farmerc[i,j,k]=1
                        if capf>a[k]
                            creditc[i,j,k]=capf-a[k]
                            collatc[i,j,k]=(1/(lambdk-1))*creditc[i,j,k]
                        else
                            creditc[i,j,k]=0
                            collatc[i,j,k]=0
                        end

                    else
                        Vnewc[i,j,k]=Ventr
                        cpolc[i,j,k]=cep_i[kprime_entr]
                        ocpolc[i,j,k]=3
                        lpolc[i,j,k]=0
                        land_rent_in_c[i,j,k]=0
                        l1polc[i,j,k]=land
                        yc[i,j,k]=znc[i]*(cape^alphae)*(hire_labor^gammae)
                        yac[i,j,k]=0
                        ync[i,j,k]=znc[i]*(cape^alphae)*(hire_labor^gammae)
                        apolc[i,j,k]=a[kprime_entr]
                        laborc[i,j,k]=hire_labor
                        capitalc[i,j,k]=cape
                        workerc[i,j,k]=0
                        farmerc[i,j,k]=0
                        if cape>a[k]
                            creditc[i,j,k]=cape-a[k]
                            collatc[i,j,k]=(1/(lambdk-1))*creditc[i,j,k]
                        else
                            creditc[i,j,k]=0
                            collatc[i,j,k]=0
                        end

                    end

                end
            end
        end

        diff=maximum(abs.(Voldc-Vnewc))
    end
    return (cpolc,  ocpolc, lpolc, land_rent_in_c,
                 yc, apolc, laborc, capitalc, workerc, farmerc,
                 yac, ync,  creditc, collatc, l1polc)


end
