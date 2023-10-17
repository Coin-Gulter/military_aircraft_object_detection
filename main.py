# Import necessary libraries
import parameters
import model

def main():
    """
    Main function of the program.

    Args: 
        None.

    Returns:
        None.
    """
    # If training, then train the model.
    h_parameters = parameters.get_config()
    page_ai = model.page_ai(h_parameters)

    print('start training')
    page_ai.train()

if __name__=="__main__":
    main()
