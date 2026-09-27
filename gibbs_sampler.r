######################################
## Gibbs Sampler for Spam Detection ##
######################################

# This code is a hierarchical Bayesian model that uses a Gibbs sampler to filter
# emails. The model assumes that the number of "spam" words in an email is
# Poisson distributed. So the inbox is a mixture of Poisson distributions with
# a latent variable of whether or not the email is spam. We are interested in
# the value of the latent variable for our classification.

# Import Packages
library(ggplot2)
library(patchwork)
library(kableExtra)

# Reading the data
spam<-read.csv("spam.csv")
n<-length(spam[,1])
real<-rep(1,n)
real[spam[,1]=="spam"]<-0


# Create a list of spam words to test each email against
spam_words<-tolower(c("Free", "entry", "wkly", "comp", "win", "final", "tkts", 
                      "Text", "entry", "FreeMsg", "darling", "fun", "XxX",
                      "entitled", "Update", "credit", "claim", "subscription",
                      "special", "pleased", "valued", "jackpot", "prize"))

# Assigning spam word counts to each email
# Create X vector representing the number of spam words in each email
X<-rep(0,n)

# Search through each email to see update spam word count
vec<-spam_words
for(word in vec){
  for(i in 1:n){
    if(grepl(word, tolower(spam[i,2]))){
      X[i]<-X[i]+1
    }
  }
}


Z0<-rbinom(n, 1, .5)        # Initial value of Z for Gibbs Sampler
beta<-1                     # Used in exponential prior
a<-1                        # Used in beta prior
b<-1                        # Used in beta Prior
m<-5000                     # Number of Gibbs Sampler iterations

# Function to make easy sampling
samp<-function(X, l0, l1, p){
  p.0<-(1-p)*dpois(X, l0)
  p.1<-p*dpois(X, l1)
  q<-p.1/(p.1+p.0)
  return(rbinom(length(X), 1, q))
}

# Running the Gibbs Sampler
## Initialize sample containers
Z<-vector(mode='list', m)
lambda<-vector("list", m)
mix_param<-vector("numeric", m)
ll<-vector("numeric", m)

## Set initial parameter values
Z[[1]]<-Z0
lambda[[1]]<-c(1,1)
mix_param[1]<-0.5
ll[1]<-sum(log(.5*dpois(X, 1)+0.5*dpois(X, 1)))

for(j in 2:m){
  ## Sample the rate and mixing parameters
  Z_j<-Z[[j-1]]
  l0<-rgamma(1,sum(X*(1-Z_j))+1, scale=beta/(beta*sum(1-Z_j)+1))
  l1<-rgamma(1,sum(X*Z_j)+1, scale=beta/(beta*sum(Z_j)+1))
  q<-rbeta(1, a+sum(Z_j), b+sum(1-Z_j))
  lambda[[j]]<-c(l0, l1)
  mix_param[j]<-q
  Z[[j]]<-samp(X, l0, l1, q)
  
  ## Perform label alignment check and switch if necessary
  if(l1>l0){
    lambda[[j]]<-rev(lambda[[j]])
    mix_param[j]<-1-mix_param[j]
    Z[[j]]<-1-Z[[j]]
  }
  
  # Calculate and track log likelihood
  ll[j]<-sum(log((1-mix_param[j])*dpois(X, lambda[[j]][1])+
                   mix_param[j]*dpois(X, lambda[[j]][2])))
}


# Finding the posterior mean of Z (post burn-in)
burnin_frac<-.75
post_idx<-floor(burnin_frac*m):m

Z.bar<-rep(0,n)
for(i in post_idx){
  Z.bar<-Z.bar+Z[[i]]
}
Z.bar<-Z.bar/length(post_idx)

# Find posterior mean of parameters
lambda_post<-do.call(rbind, lambda[post_idx])
l0_hat<-mean(lambda_post[, 1])
l1_hat<-mean(lambda_post[, 2])
q_hat<-mean(mix_param[post_idx])


# Rounding the estimates to binary spam/not-spam labels
Z_prob<-Z.bar
Z.bar<-round(Z.bar)


# Error Analysis
# Finding the accuracy of the algorithm
correct<-sum(Z.bar==real)
percent_correct<-correct/n
tabs<-table(Z.bar, real, dnn=list("Estimated", "True"))
kable(tabs, row.names = TRUE, 
      caption=paste0("Overall Accuracy: ", round(percent_correct, 3)))


# Determining the real values of the parameters of our model
l0_real<-mean(X[real==0])
l1_real<-mean(X[real==1])
q_real<-sum(real)/n



parameter_summary<-round(data.frame(Truth=c(l0_real, l1_real, q_real),
                                    Estimates=c(l0_hat, l1_hat, q_hat)),3)
rownames(parameter_summary)<-c("Lambda 0", "Lambda 1", "q")
kable(round(parameter_summary,3), row.names = TRUE, 
      caption="Model Parameters and their Estimates")

# Finding the spam word count of the mislabeled emails
missed<-which(real!=Z.bar)
l_missed<-mean(X[missed])
l_01<-(l0_real+l1_real)/2

missed_df<-round(data.frame(Count=length(missed),
                            `Avg. Spam Words in Missed Emails`=l_missed,
                            `Avg. of True Avg. Spam Words`=l_01,
                            check.names = FALSE), 3)
kable(missed_df, caption="Average Spam Words in Misclassified Emails")

# Trace plots for the parameters
L<-as.data.frame(do.call(rbind, lambda))
colnames(L)<-c("l0", "l1")
L<-cbind(Iteration=1:m, L)

((ggplot(L)+geom_line(aes(x=Iteration, y=l0))+
    labs(title="Lambda 0 trace plot")+ylab("Lambda 0")) | 
    (ggplot(L)+geom_line(aes(x=Iteration, y=l1))+
       labs(title="Trace plot for Lambda 1")+ylab("Lambda 1"))) /
  (ggplot()+geom_line(aes(x=1:m, y=mix_param))+
     labs(title="Mixing Parameter Trace Plot")+xlab("Iteration")+ylab("q"))


# Finding and plotting the error at each iteration of the Gibbs Sampler
error_vec<-c()
for(i in 1:m){
  error_vec<-c(error_vec, mean(Z[[i]]!=real))
}

for_plot<-data.frame(Iteration=1:m,
                     Error=error_vec,
                     LL=ll)

p1<-ggplot(for_plot)+geom_line(aes(x=Iteration, y=Error))+
  labs(title="Error Rate")
p2<-ggplot(for_plot)+geom_line(aes(x=Iteration, y=LL))+
  labs(title="Log Likelihood")+
  ylab("Log Likelihood")

p1 | p2



