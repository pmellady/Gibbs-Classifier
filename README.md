\newcommand\bY{\textbf{Y}}
\newcommand\bZ{\textbf{Z}}
\newcommand\bL{\boldsymbol{\lambda}}

Here is an implementation of a Gibbs sampler to email spam/not-spam data to perform classification.

## Distributional Assumptions
We start by assuming distributional properties about our data. We make the assumption that emails are drawn from a mixture of two poisson distributions. Let $Y_i$ be the number of "spam" words in the $i^{th}$ email, then we are assuming
$$
Y_i\sim (1-q)\cdot\text{pois}(\lambda_0)+q\cdot\text{pois}(\lambda_1)
$$

In this situation, the $\lambda_0$ distribution represents the count of "spam" words in spam emailsand the $\lambda_1$ distribution represents the count of "spam" words in emails that are not spam. When we say "spam" words, we mean words that are typically seen in spam email.

We introduce a latent variable, $Z_i$, which labels each email as spam or not. Using the notation that $\bL=(\lambda_0,\lambda_1)$, $\bY=(Y_1,Y_2,\cdots,Y_n)$, and $\bZ=(Z_1,\cdots,Z_n)$, we have the following hierarchical structure

$$
\begin{align*}
Y_i|q,\bL,Z_i &\sim pois(\lambda_{Z_i})\\
Z_i|q &\sim bern(q)\\
q &\sim beta(a,b)\\
\lambda_0,\lambda_1&\overset{iid}\sim exp(\beta)\quad\text{where }\beta\text{ is the exponential scale parameter}
\end{align*}
$$

## Distributional Derivations
Starting with the hierarchical model stated above, we derive each of the conditional distributions required in the model. We start with the simplest

### Distribution of $\bY|q,\bL,\bZ$
Since $Z_i$ identifies the distribution from which the email was drawn, we have that 

$$
\begin{align*}
Y_i\sim pois(\lambda_1)\text{ if }Z_i=1\\
Y_i\sim pois(\lambda_0)\text{ if }Z_i=0\\
\end{align*}
$$

### Distribution of $\lambda_i|\bY,q,\bZ$
Let us start with $\lambda_0$. We can write

$$
P(\lambda_0|\lambda_1,\bY,q,\bZ)=P(\lambda_0|\lambda_1,\bY,\bZ)=c\pi(\lambda_0)\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}f(y_i|\lambda_1)^{z_i}
$$

This is a straightforward application of Bayes' rule where the likelihood of the data is given by $\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}f(y_i|\lambda_1)^{z_i}$, the prior density is $\pi(\lambda_0)$ and the constant $c$ is for normalization. Since we assume an exponential distribution on $\lambda_0$ and since $f(y_i|\lambda_1)^{z_i}$ has no dependence on $\lambda_0$, we can simplify this as follows:
$$
P(\lambda_0|\lambda_1,\bY,q,\bZ)=c_1\frac{1}{\beta}e^{-\lambda_0/\beta}\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}=c_1\frac{1}{\beta}e^{-\lambda_0/\beta}\prod_{i=1}^n(\frac{e^{-\lambda_0}\lambda_0^{y_i}}{y_i!})^{1-z_i}
$$

We can further simplify this by rearranging more constants to obtain
$$
P(\lambda_0|\lambda_1,\bY,q,\bZ)=c_2\lambda_0^{\sum y_i(1-z_i)}e^{-\lambda_0(\frac{1}{\beta}+\sum(1-z_i))}
$$

This all tells us that 

$$
\lambda_0|\lambda_1,\bY,q,\bZ\sim gamma(\sum y_i(1-z_i)+1, (\frac{1}{\beta}+\sum(1-z_i))^{-1})
$$

Similarly, by symmetry, we have

$$
\lambda_1|\lambda_0,\bY,q,\bZ\sim gamma(\sum y_iz_i+1, (\frac{1}{\beta}+\sum z_i)^{-1})
$$

### The distribution of $q|\bY,\bZ,\bL$
Note that $P(q|\bY,\bZ,\bL)=P(q|\bZ)=p(\bZ|q)P(q)$. With the conditional distribution of $Z_i|q$ as in the model statement and with the assigned $beta(a,b)$ prior on $q$, this gives us
$$
P(q|\bY,\bZ,\bL)=cq^{a-1}(1-q)^{b-1}\prod_{i=1}^nq^{z_i}(1-q)^{1-z_i}=cq^{a+\sum z_i-1}(1-q)^{b+n-\sum z_i-1}
$$

and hence $q|\bY,\bZ,\bL\sim beta(a+\sum z_i, b+n-\sum z_i)$.

### The distribution of $\bZ|q,\bY,\bL$
Lastly, since we will be sampling the vector of $Z_i$s simultaneously, we need to find the posterior distribution of $\bZ|q,\bY,\bL$. To do this, note that
$$
P(\bZ|q,\bY,\bL)=cP(\bZ,q,\bY,\bL)=cP(\bY| q,\bZ, \bL)P(q,\bZ)=cP(\bY| q,\bZ,\bL)P(\bZ|q)P(q)
$$

Since the prior distribution of $q$ is seen as constant in the distribution of $\bZ$ is can be absorbed into the constant to obtain
$$
P(\bZ|q,\bY,\bL)=c_1P(\bY|q,\bZ,\bL)P(\bZ|q)=c_1P(\bY|\bZ,\bL)P(\bZ|q)
$$

Where we simplify $P(\bY|q,\bZ,\bL)$ to $P(\bY|\bZ,\bL)$ since the condition on $\bZ$ makes $q$ superfluous. Now, since the $Y_i$s and $Z_i$s are conditionally independent, we can write this as a product of the mass functions as follows
$$
P(\bZ|q,\bY,\bL)=c_1\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}f(y_i|\lambda_1)^{z_i}q^{z_i}(1-q)^{1-z_i}
$$

We now let $p_{0i}=f(y_i|\lambda_0)(1-q)$ and $p_{1i}=f(y_i|\lambda_1)q$, which gives
$$
P(\bZ|q,\bY,\bL)=c_1\prod_{i=1}^n p_{01}^{1-z_i}p_{1i}^{z_i}
$$

Defining $p=\frac{p_{1i}}{p_{1i}+p_{0i}}$ and simplifying the above gives
$$
P(\bZ|q,\bY,\bL)=c_2\prod_{i=1}^n (1-p)^{1-z_i}p^{z_i}=c_2\prod_{i=1}^n f(z_i|p)
$$

So, we obtain

$$
Z_i|q, \bY, \bL\sim bern(p),\quad\text{where }p=\frac{f(y_i|\lambda_1)q}{f(y_i|\lambda_1)q+f(y_i|\lambda_0)(1-q)}
$$
