run_vegan = function(method, spp_file, env_file, vars=c(), id_field="FCID",
    spp_transform="SQRT", downweighting=0)
{
	# Imports
	library(vegan)
	
	# Read in the matrices
	spp = read.csv(spp_file, row.names = id_field)
	env = read.csv(env_file, row.names = id_field)

    # Order the matrix rows on ID number
	spp = spp[order(as.numeric(rownames(spp))),]
	env = env[order(as.numeric(rownames(env))),]
	
    # Transform the species data if requested
    if (spp_transform == "SQRT") {
	    spp = sqrt(spp)
    }
    else if (spp_transform == "LOG") {
        spp = log(spp)
    }

    # If downweighting requesting, downweight the spp matrix
    if (downweighting == 1) {
        spp = downweight(spp)
    }

    # Subset the environmental variables based on the vector passed in
    env = env[,vars]

    # Create a formula string
    var_str = paste(vars, collapse=" + ")
    formula_str = paste("spp ~ ", var_str, collapse="")

    # Run the CCA model
    options(warn = -1)
    if (method == "CCA")
        ord_obj = cca(as.formula(formula_str), env)
    else if (method == "RDA")
        ord_obj = rda(as.formula(formula_str), env)
    else if (method == "DBRDA")
        ord_obj = capscale(as.formula(formula_str), env, dist="bray")

    # Return both the species matrix and the cca object 
    output = list(spp=spp, ord_obj=ord_obj)
    return(output)
}

spp_tolerance = function(ord_obj, spp_matrix) {

    # Get the LC scores and the species centroids
    num_axes = dim(ord_obj$CCA$u)[2]
    lc_scores = ord_obj$CCA$u
    species_scores = scores(ord_obj, display="species", choices=c(1:num_axes))
    
    # Set up the xiuk array
    xiuk = array(NA, 
        dim=c(nrow(lc_scores), ncol(lc_scores), nrow(species_scores)),
        dimnames=list(rownames(lc_scores), colnames(lc_scores), 
            rownames(species_scores)))
    
    # Fill the xiuk array
    for (i in rownames(species_scores)) {
        species_row = t(apply(lc_scores, 1, function(x) {
            x - species_scores[i,]
        }))
        xiuk[,,i] = species_row
    }
    
    # Square the xiuk matrix
    xiuk_sqr = xiuk * xiuk
    
    # Get the column sums of the spp_matrix
    spp_sums = colSums(spp_matrix)
    
    # Transpose the species data
    t_spp = t(spp_matrix)
    
    # Set up the y_xiuk array
    num_species = dim(xiuk_sqr)[3]
    num_axes = dim(xiuk_sqr)[2]
    y_xiuk = array(NA, dim=c(num_species, num_axes),
      dimnames =list(rownames(species_scores), colnames(xiuk_sqr)))
    
    # Fill the y_xiuk array
    for (i in rownames(y_xiuk)) {
        y_xiuk[i,] = t_spp[i,] %*% xiuk_sqr[,,i]
    }
    
    # Calculate tolerances
    normalized_y_xiuk = y_xiuk / spp_sums
    tolerance = sqrt(normalized_y_xiuk)
}

write_vegan = function(method, spp_file, env_file, vars=c(), id_field="FCID",
    spp_transform="SQRT", downweighting=0, out_file="vegan.txt")
{
    # Read in the vegan package
    require(vegan)

    # Run the ordination and return the species matrix and ordination object
    output = 
        run_vegan(method, spp_file, env_file, vars=vars, id_field=id_field,
            spp_transform=spp_transform, downweighting=downweighting)
    spp = output$spp
    ord_obj = output$ord_obj

    # Eigenvalues 
    # ord_obj$CCA$eig
    write("### Eigenvalues ###", out_file, append=F)
    x = format(ord_obj$CCA$eig, scientific=F, trim=T, digits=6)
    write.table(x, out_file, col.names=F, row.names=T, quote=F, append=T, 
        sep=",")
    write("", out_file, append=T)
  
    # Ordination Variable means 
    # ord_obj$CCA$envcentre
    write("### Variable Means ###", out_file, append=T)
    x = format(ord_obj$CCA$envcentre, scientific=F, trim=T, digits=6)
    write.table(x, out_file, col.names=F, row.names=T, quote=F, append=T, 
        sep=",") 
    write("", out_file, append=T)
 
    # Coefficient loadings for explanatory data that go into model
    # coef(ord_obj)
    write("### Coefficient Loadings ###", out_file, append=T)
    VARIABLE = rownames(coef(ord_obj, quiet=T))
    x = data.frame(VARIABLE, format(coef(ord_obj, quiet=T), scientific=F,
        trim=T, digits=6))
    write.table(x, out_file, row.names=F, col.names=T, quote=F, append=T, 
        sep=",")  
    write("", out_file, append = T)

    # Biplot Scores
    write("### Biplot Scores ###", out_file, append = T)
    biplot.axes = summary(ord_obj, axes=ord_obj$CCA$rank)$biplot
    # biplot.axes = biplot.axes[,grep("CCA", colnames(biplot.axes))]
    VARIABLE = rownames(biplot.axes)
    x = data.frame(VARIABLE, format(biplot.axes, scientific=F, 
        trim=T, digits=6))
    write.table(x, out_file, row.names=F, col.names=T, quote=F, append=T, 
        sep=",") 
    write("", out_file, append = T)

    # Species centroids
    write("### Species Centroids ###", out_file, append = T)
    num_axes = dim(ord_obj$CCA$u)[2]
    species_scores = scores(ord_obj, display="species", choices=c(1:num_axes))
    SPECIES = rownames(species_scores)
    x = data.frame(SPECIES, format(species_scores, scientific=F, trim=T,
        digits=6))
    write.table(x, out_file, col.names=T, row.names=F, quote=F, append=T, 
       sep=",")
    write("", out_file, append = T)

    # Species tolerances
    write("### Species Tolerances ###", out_file, append = T)
    tolerances = spp_tolerance(ord_obj, spp)
    SPECIES = rownames(tolerances)
    x = data.frame(SPECIES, format(tolerances, scientific=F, trim=T, digits=6))
    write.table(x, out_file, col.names=T, row.names=F, quote=F, append=T, 
        sep=",")
    write("", out_file, append = T)

    # Species weights, N2, etc.
    write("### Miscellaneous Species Information ###", out_file, append=T)
    species = rownames(ord_obj$CCA$v)
    weight = colSums(spp)
    n_2 = 1.0 / colSums(t(apply(spp, 1, "/", weight) ^ 2))
    out = cbind(weight, n_2)
    colnames(out) = c("WEIGHT", "N2")
    SPECIES = rownames(out)
    x = data.frame(SPECIES, format(out, scientific=F, trim=T, digits=6))
    write.table(x, out_file, col.names=T, row.names=F, quote=F, append=T, 
        sep=",")
    write("", out_file, append = T)

    # Site scores which are linear combinations of environmental variables
    # ord_obj$CCA$u
    write("### Site LC Scores ###", out_file, append=T)
    ID = rownames(ord_obj$CCA$u)
    x = data.frame(ID, format(ord_obj$CCA$u, scientific=F, trim=T, digits=6))
    write.table(x, out_file, col.names=T, row.names=F, quote=F, append=T, 
        sep=",")
    write("", out_file, append = T)

    # Site scores that are derived from weighted averaging
    # ord_obj$CCA$wa
    write("### Site WA Scores ###", out_file, append=T)
    ID = rownames(ord_obj$CCA$wa)
    x = data.frame(ID, format(ord_obj$CCA$wa, scientific=F, trim=T, digits=6))
    write.table(x, out_file, col.names=T, row.names=F, quote=F, append=T, 
        sep=",")
    write("", out_file, append = T)

    # Site weights, N2, etc.
    write("### Miscellaneous Site Information ###", out_file, append=T)
    weight = rowSums(spp)
    n_2 = 1.0 / rowSums((spp / weight) ^ 2)
    out = cbind(weight, n_2)
    colnames(out) = c("WEIGHT", "N2")
    ID = rownames(out)
    x = data.frame(ID, format(out, scientific=F, trim=T, digits=6))
    write.table(x, out_file, col.names=T, row.names=F, quote=F, append=T, 
        sep=",")
}
