library(ggplot2)
library(reshape2)
library(tikzDevice)
library(stringr)
print('creating diagrams...')
args = commandArgs(trailingOnly=TRUE)

dat.inner  <- read.csv(args[1], header=TRUE)

do.plot <- function(dat, dataset) {
    #dat.molten <- melt(dat, id.vars=c("Category"))
    dat.molten <- dat
    g <- ggplot(dat.molten, aes(x=Category, y=value, fill=estimator)) +
        facet_grid(metric~estimator) + geom_bar(stat="identity") +
        ggtitle(str_c("Dataset: ", dataset)) + xlab("Category") + ylab("Value [Percent]") +
        theme(legend.position="top")
    return(g)
}

g <- do.plot(dat.inner, args[3])
print(g) ## Show interactively
                                        
## Save as TikZ drawing
# tikz(paste(args[2],"\\tex\\",args[3],".tex",sep=""), width=10, height=12)
# print(g)
# dev.off()

## Save as PDF
ggsave(paste(args[2],"\\pdf\\",args[3],".pdf",sep=""), g,device="pdf", width=10, height=10)

print('done')