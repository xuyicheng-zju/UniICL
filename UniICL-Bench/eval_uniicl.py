from eval_bagel import *  # noqa: F401,F403
from eval_bagel import SafeInterleaveInferencer, load_bagel_model, main


load_uniicl_model = load_bagel_model
SafeUniICLInferencer = SafeInterleaveInferencer


if __name__ == "__main__":
    main()
