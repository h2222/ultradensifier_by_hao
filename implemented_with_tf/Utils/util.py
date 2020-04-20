# coding=utf-8
import scipy.stats as st







# use to final evaluation
def evall(true, prediction, metric='r'):
        '''
        Expects pandas data frames.
        '''
        metrics={'r':lambda x,y:st.pearsonr(x,y)[0],
                 'rmse':rmse,
                 'mae':mae
                }
        metric=metrics[metric]
        row=[]
        for var in list(prediction):
                value=metric(prediction[var], true[var])
                row+=[value]
        return row


def average_results_df(results_df):
        avg=results_df.mean(axis=0)
        sd=results_df.std(axis=0)
        results_df.loc['Average']=avg
        results_df.loc['SD']=sd
        results_df['Average']=results_df.mean(axis=1)
        return results_df

