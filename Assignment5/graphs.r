#load in the libraries
library(tidyverse)

#read the data
data <- read.csv("data.csv")

library(tidyverse)

#clean it a bit for graphing
data_long <- data %>%
  pivot_longer(cols = c(train_acc, test_acc),
               names_to = "type",
               values_to = "accuracy")

data_long <- data_long %>%
  mutate(trial = paste0("Trial ", trial))

#plot the data
ggplot(data_long, aes(x = epoch,
                   y = accuracy,
                   color = type)) +
  geom_line(size = 1) +
  facet_grid(trial ~ cell_type, switch = "y") +
  labs(
    x = "Epoch",
    y = "Accuracy",
    color = "Accuracy Type",
    title = "Accuracy Across Trial Measures"
  ) +
  theme_bw()
