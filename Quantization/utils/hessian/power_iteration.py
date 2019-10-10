import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils.hessian.utils import group_product, de_variable, get_blocks, get_max_error


def power_iteration(
        train_dataloader,
        model,
        blocktype,
        data_percentage,
        train_batch_size,
        device,
        get_loss,
        relative_tolerance=1e-4,
        seed=0):
    '''
    Using power iteration to calculate the largest eigenvalues for the specific blocks.

    Inputs:
        train_dataloader (class:`torch.utils.data.DataLoader`): 
            the input dataloader. NOTE: DO NOT include any randomness in the dataloader.
        model (class:`nn.Module`): a neural network model.
        blocktype (class: `nn.Module`): The type of the blocks that will calculate largest eigenvalues.
        data_percentage (class: `float`): the percentage of the data used to do power iteration.
            data_percentage should be within (0,1]
        train_batch_size: the training batch size
        device (class: `str`): the device type, can be gpu or cpu.
        get_loss (class: function): 
            a function to get the loss based on input batch data.  
            Examples:
                for image classification:
                	```
    	            def get_loss(batch):
                        batch = tuple(t.to(device) for t in batch)
                        inputs, targets = batch
                        outputs = model(inputs)
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, targets)
                        return loss
                    ```
                for GLUE NLP example:
                	```
                	def get_loss(batch):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, input_mask, segment_ids, label_ids = batch
                        logits = model(input_ids, segment_ids, input_mask, labels=None)
                        if output_mode == "classification":
                            loss_fct = CrossEntropyLoss()
                            loss = loss_fct(
                                logits.view(-1, num_labels), label_ids.view(-1))
                        elif output_mode == "regression":
                            loss_fct = MSELoss()
                            loss = loss_fct(logits.view(-1), label_ids.view(-1))

                        return loss
                    ```
        relative_tolerance (class: 'float'): the relative tolerance for stopping power iteration.
        seed (class: 'int'): the random seed number.

    Returns:
        A list containing all the names whose eigenvalues are calculated.
        A list containing all the converged eigenvalues.
    '''

    assert(data_percentage>0 and data_percentage<=1)
  
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    percentage_index = len(train_dataloader.dataset) * \
        data_percentage / train_batch_size
    print(f'percentage_index: {percentage_index}')

    model.to(device)
    model.eval()

    block_names, model_blocks = get_blocks(model, blocktype)

    # initialize the random vectors
    vectors = []
    for model_block in model_blocks:
        v = [torch.randn(p.size()).to(device) for p in model_block.parameters()]
        v = de_variable(v)
        vectors.append(v)

    lambda_old = np.zeros(len(vectors))
    lambdas = np.ones(len(vectors))
    i = 0

    while (get_max_error(lambdas, lambda_old) >= relative_tolerance):

        lambda_old = np.copy(lambdas)
        accumulate_Hvs = []
        for model_block in model_blocks:
            accumulate_Hv = [torch.zeros(p.size()).to(device) for p in model_block.parameters()]
            accumulate_Hvs.append(accumulate_Hv)
           	
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if step >= percentage_index:
                break
            loss = get_loss(batch)
            loss.backward(create_graph=True)

            for block_number, model_block in enumerate(model_blocks):
                grads = [param.grad for param in model_block.parameters()]
                params = model_block.parameters()

                Hv = torch.autograd.grad(
                    grads,
                    params,
                    grad_outputs=vectors[block_number],
                    only_inputs=True,
                    retain_graph=True)
                accumulate_Hvs[block_number] = [
                        acc_Hv_p + Hv_p for acc_Hv_p, Hv_p in zip(accumulate_Hvs[block_number], Hv)
                    ]

            model.zero_grad()
        for block_number, model_block in enumerate(model_blocks):
        	# calculate raylay quotients
            lambdas[block_number] = group_product(accumulate_Hvs[block_number], vectors[block_number]).item() / percentage_index
            vectors[block_number] = de_variable(accumulate_Hvs[block_number])

        i += 1

        eigenvalues = {'layer_name':block_names,'eigenvalues':lambdas}
        eigenvalues_df = pd.DataFrame(eigenvalues)

        print(f'power iteration at iteration {i}, the largest eigenvalues are ')
        print(eigenvalues_df)

    print(f'power iteration at iteration {i}, the largest eigenvalue is converged.')
    print(eigenvalues_df)

    return eigenvalues_df

