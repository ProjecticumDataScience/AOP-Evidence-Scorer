## model_train_pytorch
###  This script teaches the model to identify causal relationships based on a given dataset. 
This script teaches the model to identify causal relationships based on a given dataset. For example:
nuclear_receptor_changes_pparg->de_novo_lipogenesis_fa_synthesis

When we send this data to the model, we first encode it,
- enc_up = LabelEncoder().fit(df["KE_upstream"])
- enc_down = LabelEncoder().fit(df["KE_downstream"])

and then we save it in the endocerns.

- pickle.dump(enc_up, open("models/enc_up.pkl","wb"))
- pickle.dump(enc_down, open("models/enc_down.pkl","wb"))


We get one big model in .pt format in the "models" folder, 
along with several smaller endocores (upstream, downstream, context)

The .pt extension is  used format for saving PyTorch models,model's
weights and architectures (learned parameters as tensors)
