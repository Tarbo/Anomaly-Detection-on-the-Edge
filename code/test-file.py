#%%
import pickle
import pyre
import os


def node(name):
    name = name + '-' + str(os.getpid())
    agent = pyre.Pyre(name)
    agent.join('Ezeme')
    agent.start()
    try:

        events = agent.events()
        while True:
            if events:
                event = next(events)
                print(f'>>> MSG TYPE: {event.type}')
                if event.type == "WHISPER":
                    msg = pickle.loads(event.msg[0])
                    print(msg['name'])
    except Exception as identifier:
        print(f'>>> Exception Type: {identifier}')
        agent.stop()


#%%
