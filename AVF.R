
AVF <- function(input,output){
  library(plyr);
  mydata <- read.csv(input, header = FALSE, sep = ",",stringsAsFactors=,encoding = "utf-8")
  #print(head(mydata))
  row <- nrow(mydata)
  col <- ncol(mydata)

  for ( i in 1: row)
  {

    sum <- 0
    for (j in 1:col)
    {
      #print(j)
      freq_table = table(unlist(unname(mydata[,j])))

      value <- paste0(mydata[i,j],"")

      freq <- freq_table [value]
      #print(paste0(value,freq))
      sum <- sum + freq

    }

    mydata[i,"Score"] <- sum/col
    mydata[i,"Score2"] <- paste0(sum,"/" ,col)
  }
  write.csv(mydata,output)
}