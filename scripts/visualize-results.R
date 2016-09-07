####
# visualize results
###

library(ggplot2)

setwd("~/Documents/microbiome-regression/")
results.files = list.files("./results/LS/10-folds/", full.names = TRUE)

for (i in c(1:length(results.files))){
  df = read.csv(results.files[i], header = TRUE, stringsAsFactors = FALSE)
  df$tax.level = sapply(df$data, FUN = function(x) strsplit(strsplit(x, split = "LS-")[[1]][2], split = "-biomarkers")[[1]][1])
  
  title.name = strsplit(strsplit(results.files[i], split= "//")[[1]][2], ".csv")[[1]][1]
  ggplot(df, aes(x = tax.level, y = Mean.score, colour = model)) +
    geom_errorbar(aes(ymin=Mean.score-STD.score/10, ymax=Mean.score+STD.score/10), width=.1) +
    geom_point(size = 4) + ggtitle(title.name) + ylab("Mean absolute error") + xlab("")
  ggsave(paste0("./results/", title.name, ".pdf"))
}
