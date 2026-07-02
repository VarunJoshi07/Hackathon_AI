def calculate_hiring_likelihood(df):
    signals = ['open_to_work', 'response_likelihood', 'recent_activity']
    likelihood_score = df.reindex(columns=signals, fill_value=0).mean(axis=1)
    return likelihood_score.values