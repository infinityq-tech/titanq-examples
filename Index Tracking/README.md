# README

#### This is an example of using TitanQ to solve real life index tracking problems.
--------------------------------------------------------------------------------

## Index Tracking 

The index tracking problem in finance refers to the challenge of creating and managing a portfolio that closely replicates the performance of a specific stock market index, such as the S&P 500 or FTSE 100, while minimizing tracking error and associated costs. This problem is central to passive investment strategies, such as index funds and exchange-traded funds (ETFs), which aim to deliver the same returns as the index they track.

This problem also arises when a portfolio manager may be interested in mimicking the behavior of an asset that they cannot hold. For example, a French asset manager for a pension fund may not be allowed to invest in the US or Japanese stock exchange. Instead, they would like to buy a basket of stocks that mimics or tracks the performance of this outside portfolio. 

We analyze the Index Tracking problem as a non-linear quadratically constrained, quadratic optimization problem (MIQCP). This problem is challenging due to the non-linearity and the constraints, making it computationally intensive and difficult to solve directly especially when taking into account real world constraints and a large number of possible assets. 



# TitanQ SDK

The required packages are listed in *requirements.txt* and can be installed using pip:

```bash
pip install -r requirements.txt
```

## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.
