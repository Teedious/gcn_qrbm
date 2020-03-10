library(ggplot2)
library(reshape2)
library(tikzDevice)
library(stringr)

dat.random_20000_20000_200_0p1_5_metrics_  <- read.csv("./random_20000_20000_200_0p1_5_metrics_.txt", header=TRUE)

do.plot <- function(dat, dataset) {
    dat.molten <- melt(dat, id.vars=c("Category"))
    g <- ggplot(dat.molten, aes(x=Category, y=value, fill=variable, colour=variable)) +
        facet_wrap(~variable, scales="free_y", ncol=4) + geom_bar(stat="identity") +
        ggtitle(str_c("Dataset: ", dataset)) + xlab("Category") + ylab("Value [Percent]") +
        theme(legend.position="top")
    return(g)
}

g <- do.plot(dat.random_20000_20000_200_0p1_5_metrics_, "random_20000_20000_200_0p1_5_metrics_")
print(g) ## Show interactively
                                        
## Save as TikZ drawing
tikz("./tmp/random_20000_20000_200_0p1_5_metrics_.tex", width=10, height=12)
print(g)
dev.off()

## Save as PDF
ggsave("./tmp/random_20000_20000_200_0p1_5_metrics_.pdf", g, width=10, height=10)
