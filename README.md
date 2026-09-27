This repository contains a derivation and implementation of an interesting classification algorithm. By making the assumption that spam emails contain "spam" words, we can calculate the total spam word count per email and assume that these counts are Poisson distributed. Not all emails are spam, though, and some of those non-spam emails may contain "spam" words, so we can conceptualize an email inbox as a random sample from a mixture of Poisson distributions. The following derivation is an implementation of a Gibbs sampler specifically for this set-up.

## Distributional Assumptions
We start by assuming distributional properties about our data. We make the assumption that emails are drawn from a mixture of two poisson distributions. Let $Y_i$ be the number of "spam" words in the $i^{th}$ email, then we are assuming

$$
Y_i\sim (1-q)\cdot\text{pois}(\lambda_0)+q\cdot\text{pois}(\lambda_1)
$$

In this situation, the $\lambda_0$ distribution represents the count of "spam" words in spam emails and the $\lambda_1$ distribution represents the count of "spam" words in emails that are not spam. When we say "spam" words, we mean words that are typically seen in spam email.

We introduce a latent variable, $Z_i$, which labels each email as spam or not. Using the notation that $\boldsymbol{\lambda}=(\lambda_0,\lambda_1)$, $\textbf{Y}=(Y_1,Y_2,\cdots,Y_n)$, and $\textbf{Z}=(Z_1,\cdots,Z_n)$, we have the following hierarchical structure

$$
\begin{align*}
Y_i|q,\boldsymbol{\lambda},Z_i &\sim pois(\lambda_{Z_i})\\
Z_i|q &\sim bern(q)\\
q &\sim beta(a,b)\\
\lambda_0,\lambda_1&\overset{iid}\sim exp(\beta)\quad\text{where }\beta\text{ is the exponential scale parameter}
\end{align*}
$$

## Distributional Derivations
Starting with the hierarchical model stated above, we derive each of the conditional distributions required in the model. We start with the simplest

### Distribution of $\textbf{Y}|q,\boldsymbol{\lambda},	\textbf{Z}$
Since $Z_i$ identifies the distribution from which the email was drawn, we have that 

$$
\begin{align*}
Y_i\sim pois(\lambda_1)\text{ if }Z_i=1\\
Y_i\sim pois(\lambda_0)\text{ if }Z_i=0\\
\end{align*}
$$

### Distribution of $\lambda_i|	\textbf{Y},q,	\textbf{Z}$
Let us start with $\lambda_0$. We can write

$$
P(\lambda_0|\lambda_1, \textbf{Y},q, \textbf{Z})=P(\lambda_0|\lambda_1,	\textbf{Y},	\textbf{Z})=c\pi(\lambda_0)\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}f(y_i|\lambda_1)^{z_i}
$$

This is a straightforward application of Bayes' rule where the likelihood of the data is given by $\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}f(y_i|\lambda_1)^{z_i}$, the prior density is $\pi(\lambda_0)$ and the constant $c$ is for normalization. Since we assume an exponential distribution on $\lambda_0$ and since $f(y_i|\lambda_1)^{z_i}$ has no dependence on $\lambda_0$, we can simplify this as follows:

$$
P(\lambda_0|\lambda_1,	\textbf{Y},q,	\textbf{Z})=c_1\frac{1}{\beta}e^{-\lambda_0/\beta}\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}=c_1\frac{1}{\beta}e^{-\lambda_0/\beta}\prod_{i=1}^n(\frac{e^{-\lambda_0}\lambda_0^{y_i}}{y_i!})^{1-z_i}
$$

We can further simplify this by rearranging more constants to obtain

$$
P(\lambda_0|\lambda_1,	\textbf{Y},q,	\textbf{Z})=c_2\lambda_0^{\sum y_i(1-z_i)}e^{-\lambda_0(\frac{1}{\beta}+\sum(1-z_i))}
$$

This all tells us that 

$$
\lambda_0|\lambda_1,	\textbf{Y},q,	\textbf{Z}\sim gamma(\sum y_i(1-z_i)+1, (\frac{1}{\beta}+\sum(1-z_i))^{-1})
$$

Similarly, by symmetry, we have

$$
\lambda_1|\lambda_0,	\textbf{Y},q,	\textbf{Z}\sim gamma(\sum y_iz_i+1, (\frac{1}{\beta}+\sum z_i)^{-1})
$$

### The distribution of $q|	\textbf{Y},	\textbf{Z},\boldsymbol{\lambda}$
Note that $P(q|	\textbf{Y},	\textbf{Z},\boldsymbol{\lambda})=P(q|	\textbf{Z})=p(\textbf{Z}|q)P(q)$. With the conditional distribution of $Z_i|q$ as in the model statement and with the assigned $beta(a,b)$ prior on $q$, this gives us

$$
P(q|\textbf{Y},	\textbf{Z},\boldsymbol{\lambda})=cq^{a-1}(1-q)^{b-1}\prod_{i=1}^nq^{z_i}(1-q)^{1-z_i}=cq^{a+\sum z_i-1}(1-q)^{b+n-\sum z_i-1}
$$

and hence $q|	\textbf{Y},	\textbf{Z},\boldsymbol{\lambda}\sim beta(a+\sum z_i, b+n-\sum z_i)$.

### The distribution of $\textbf{Z}|q,\textbf{Y},\boldsymbol{\lambda}$
Lastly, since we will be sampling the vector of $Z_i$s simultaneously, we need to find the posterior distribution of $\textbf{Z}|q,\textbf{Y},\boldsymbol{\lambda}$. To do this, note that

$$
P(\textbf{Z}|q,\textbf{Y},\boldsymbol{\lambda})=cP(	\textbf{Z},q,	\textbf{Y},\boldsymbol{\lambda})=cP(	\textbf{Y}| q,	\textbf{Z}, \boldsymbol{\lambda})P(q,	\textbf{Z})=cP(	\textbf{Y}| q,	\textbf{Z},\boldsymbol{\lambda})P(	\textbf{Z}|q)P(q)
$$

Since the prior distribution of $q$ is seen as constant in the distribution of $	\textbf{Z}$ is can be absorbed into the constant to obtain

$$
P(	\textbf{Z}|q,	\textbf{Y},\boldsymbol{\lambda})=c_1P(	\textbf{Y}|q,	\textbf{Z},\boldsymbol{\lambda})P(	\textbf{Z}|q)=c_1P(	\textbf{Y}|	\textbf{Z},\boldsymbol{\lambda})P(	\textbf{Z}|q)
$$

Where we simplify $P(	\textbf{Y}|q,	\textbf{Z},\boldsymbol{\lambda})$ to $P(	\textbf{Y}|	\textbf{Z},\boldsymbol{\lambda})$ since the condition on $	\textbf{Z}$ makes $q$ superfluous. Now, since the $Y_i$s and $Z_i$s are conditionally independent, we can write this as a product of the mass functions as follows

$$
P(	\textbf{Z}|q,	\textbf{Y},\boldsymbol{\lambda})=c_1\prod_{i=1}^nf(y_i|\lambda_0)^{1-z_i}f(y_i|\lambda_1)^{z_i}q^{z_i}(1-q)^{1-z_i}
$$

We now let $p_{0i}=f(y_i|\lambda_0)(1-q)$ and $p_{1i}=f(y_i|\lambda_1)q$, which gives

$$
P(	\textbf{Z}|q,	\textbf{Y},\boldsymbol{\lambda})=c_1\prod_{i=1}^n p_{01}^{1-z_i}p_{1i}^{z_i}
$$

Defining $p=\frac{p_{1i}}{p_{1i}+p_{0i}}$ and simplifying the above gives

$$
P(	\textbf{Z}|q,	\textbf{Y},\boldsymbol{\lambda})=c_2\prod_{i=1}^n (1-p)^{1-z_i}p^{z_i}=c_2\prod_{i=1}^n f(z_i|p)
$$

So, we obtain

$$
Z_i|q, 	\textbf{Y}, \boldsymbol{\lambda}\sim bern(p),\quad\text{where }p=\frac{f(y_i|\lambda_1)q}{f(y_i|\lambda_1)q+f(y_i|\lambda_0)(1-q)}
$$

## Gibbs Sampling Loop
Now that we have the necessary conditional posteriors, we can implement a Gibbs sampler in the following order:

* Sample $\lambda_0|\textbf{Y},\textbf{Z},\lambda_1, q$
* Sample $\lambda_1|\textbf{Y},\textbf{Z},\lambda_0, q$
* Sample $q|\textbf{Y},\textbf{Z}, \lambda_0, \lambda_1$
* Sample $\textbf{Z}|\textbf{Y}, \lambda_0, \lambda_1, q$

We additionally impose an ordering constraint on the $\lambda_i$ values to prevent label switching while sampling. If, at any point in our sampling, we obtain $\lambda_1>\lambda_0$, we will switch every parameter to the opposite class. We impose this constraint by using the assumption that spam emails should have the higher average spam word count. This constraint is just the mathematical translation of this statement.
