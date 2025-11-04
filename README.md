### Efficient Scaling

This aims to be a repository to have quick methodologies to build and fit scaling laws, given data.

The methods here are heavily based on Chinchilla ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)), and its three approaches.

For each of the approaches, we expect a dataframe with specific column names that represent certain entities. 

Typically, for the most general methodology, we expect a flattened data (usually a `pd.DataFrame`) with repeated rows for columns such as model sizes, hyperparameters, or broader training types.

For each approach, we expect suitable filtering and subsetting of this data, conditioned on the approach being used for the scaling law fit.

As a *minimal requirement*, we need a `parameters (N)` column, a `tokens/samples (D)` column, along with the final loss per `(N, D)`.