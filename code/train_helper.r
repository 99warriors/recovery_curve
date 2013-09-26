library(functional)

read_in_pops <- function(file){
  d <- read.csv(file, header=F)
  return(list(pop_a=d[1,1], pop_b=d[1,2], pop_c=d[1,3]))
}

read_in_hypers <- function(file){
  d <- read.csv(file, header=F)
  return(list(c_a=d[1,1],c_b=d[1,2],c_c=d[1,3],l_a=d[1,4],l_b=d[1,5],l_c=d[1,6],l_m=d[1,7]))
}

# function that takes in pid and returns long thin vector of its times/values
get_tvs_given_folder <- function(pid, folder){
    file <- paste(folder,'/',pid,sep='')
    #print(file)
    #print(folder)
    #print(pid)
    #print('ZZZZZ')
    ans <- read.csv(file,header=F)
    ans <- ans[complete.cases(ans),]
    #print(ans)
    return(ans)
}

# function that takes in a pid, and returns a vector of its covariates
get_x_by_pid_given_xs <- function(pid, xs){
    return(xs[pid,])		     
}

# function that takes in vector, applies function that returns a matrix, and rbinds all of those matricies
cbind_apply <- function(v, f){
    return(Reduce(rbind,lapply(v, f)))	    
}

# function that takes in pid and returns how many times it has
get_tv_length <- function(pid, get_tvs_f){
    return(dim(get_tvs_f(pid))[1])
}

# reads data stuff besdies hyperparams and pops
get_real_full_data_diffcovs <- function(folder_path){
    ss_file <- paste(folder_path,'ss.csv',sep='/')		
    ss <- read.csv(ss_file,header=F,row.names=1)[,1]
	
    xas_file <- paste(folder_path,'xas.csv',sep='/')
    print('xas')
    print(xas_file)	
    xas <- read.csv(xas_file,header=T,row.names=1)

    xbs_file <- paste(folder_path,'xbs.csv',sep='/')
    print('xbs')
    print(xbs_file)		
    xbs <- read.csv(xbs_file,header=T,row.names=1)

    xcs_file <- paste(folder_path,'xcs.csv',sep='/')	
    print('xcs')
    print(xcs_file)	
    xcs <- read.csv(xcs_file,header=T,row.names=1)
			    
    pids_file <- paste(folder_path, 'pids.csv',sep='/')
    print('pids')
    print(pids_file)	
    pids <- read.csv(pids_file, header=F)[,1]
    print(pids)
    print('after')

    #get_x_by_pid <- Curry(get_x_by_pid_given_xs, xs=xs_a)
    get_tvs <- Curry(get_tvs_given_folder, folder=paste(folder_path,'/','datapoints',sep=''))
    tvs <- cbind_apply(pids, get_tvs)
    tv_lengths <- sapply(pids, Curry(get_tv_length, get_tvs_f=get_tvs))

    data <- list(ls=tv_lengths,ts=tvs[,1],vs=tvs[,2],xas=xas,xbs=xbs,xcs=xcs,ss=ss,N=dim(xas)[1],K_A=dim(xas)[2],K_B=dim(xbs)[2],K_C=dim(xcs)[2],L=dim(tvs)[1])

    return(data)

}

# given a fitted model and folder, writes the posterior parameters in the folder
write_full_posterior_parameters <- function(fit, folder){

    results <- extract(fit)

    B_aR <- results$B_a
    B_bR <- results$B_b
    B_cR <- results$B_c
    phi_aR <- results$phi_a
    phi_bR <- results$phi_b
    phi_cR <- results$phi_c
    phi_mR <- results$phi_m

    write.table(B_aR, paste(folder,'/','out_B_a.csv',sep=''),quote=F,row.names=F,col.names=F,sep=',')
    write.table(B_bR, paste(folder,'/','out_B_b.csv',sep=''),quote=F,row.names=F,col.names=F,sep=',')
    write.table(B_cR, paste(folder,'/','out_B_c.csv',sep=''),quote=F,row.names=F,col.names=F,sep=',')
    print('phis')
    write.table(phi_aR, paste(folder,'/','out_phi_a.csv',sep=''),quote=F,row.names=F,col.names=F,sep=',')
    write.table(phi_bR, paste(folder,'/','out_phi_b.csv',sep=''),quote=F,row.names=F,col.names=F,sep=',')
    write.table(phi_cR, paste(folder,'/','out_phi_c.csv',sep=''),quote=F,row.names=F,col.names=F,sep=',')
    write.table(phi_mR, paste(folder,'/','out_phi_m.csv',sep=''),quote=F,row.names=F,col.names=F,sep=',')

}