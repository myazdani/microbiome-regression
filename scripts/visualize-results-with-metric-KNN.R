###
#visualize-results with-metric-KNN
##

library(ggplot2)

setwd("~/Documents/microbiome-regression/")
results.files = list.files("./results/LS/absolute_mean_errors", full.names = TRUE)

results.metric = list.files("./results/LS", full.names = TRUE, pattern = "*.csv")
i = 1

for (i in c(1:length(results.metric))){
  file.key = strsplit(results.metric[i], split = "MetricKNN_")[[1]][2]
  file.match = grep(paste0("/", file.key), results.files)
  df.1 = read.csv(results.metric[i], header = TRUE, stringsAsFactors = FALSE)
  df.2 = read.csv(results.files[file.match], header = TRUE, stringsAsFactors = FALSE)
  df = rbind(df.1, df.2)
  df$tax.level = sapply(df$data, FUN = function(x) strsplit(strsplit(x, split = "LS-")[[1]][2], split = "-biomarkers")[[1]][1])
  title.name = strsplit(strsplit(results.metric[i], split= "MetricKNN_")[[1]][2], ".csv")[[1]][1]
  
  ggplot(df, aes(x = tax.level, y = Mean.score, colour = model)) +
    geom_errorbar(aes(ymin=Mean.score-STD.score/5, ymax=Mean.score+STD.score/5), width=.1) +
    geom_point(size = 4) + ggtitle(title.name) + ylab("Mean absolute error") + xlab("")
  ggsave(paste0("./results/figures/", title.name, ".pdf"))
}
