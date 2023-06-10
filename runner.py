import torch
from model import QTrainer, DQNRegular, ResNet, ResidualBlock, DQNRegular, DQN
from agent import test_pretimed, train, test, Agent

LR = 0.001
without_ev = False

traintest = int(input("1.Train 2.Test"))
if traintest == 1:
    m = int(input("Model - 1.CNN_DQN 2.Resnet_DQN 3.ContourApproximation_DQN"))
    if m == 1:
        agent = Agent()
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        agent.Tensor = FloatTensor
        agent.model = DQN(4)
        agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
        train(agent)
    elif m == 2:
        agent = Agent()
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        agent.Tensor = FloatTensor
        agent.model = ResNet(ResidualBlock, [2, 2, 2, 2], 4)
        agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
        train(agent, is_resnet=True)
    elif m == 3:
        agent = Agent()
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        agent.Tensor = FloatTensor
        agent.model = DQNRegular(4, 256, 4)
        agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
        train(agent, state_type = "tuple")
    else:
        print("Invalid input")
        exit()

elif traintest == 2:
    m = int(input("Model - 1.CNN_DQN 2.Resnet_DQN 3.ContourApproximation_DQN 4.Pre-timed"))
    if m == 1:
        agent = Agent()
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        agent.Tensor = FloatTensor
        agent.model = DQN(4)
        agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
        test(agent, "models_DQN", without_ev=without_ev)
    elif m == 2:
        agent = Agent()
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        agent.Tensor = FloatTensor
        agent.model = ResNet(ResidualBlock, [2, 2, 2, 2], 4)
        agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
        test(agent, "models_Resnet", is_resnet=True, without_ev=without_ev)
    elif m == 3:
        agent = Agent()
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        agent.Tensor = FloatTensor
        agent.model = DQNRegular(4, 256, 4)
        agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
        test(agent, "models_DQNRegular", state_type = "tuple", without_ev=without_ev)
    elif m == 4:
        agent = Agent()
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        agent.Tensor = FloatTensor
        agent.model = DQN(4)
        agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
        test_pretimed(agent)        
    else:
        print("Invalid input")
        exit()

else:
    print("Invalid input")
    exit()
    
    agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)        
    train(agent)    


test_pretimed(agent)
