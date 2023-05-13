import gradio as gr
from system import*
from flask_cors import CORS

fluid_options = ['CO2', 'N2O']



demo1 = gr.Interface(
  fn=plot_system,
  inputs=[
      gr.inputs.Slider(minimum=0.5, maximum=1, default=0.7, label='分流比'),
      gr.inputs.Number(default=20, label='最大压强 (MPa)'),
      gr.inputs.Number(default=900, label='最高温度 (K)'),
      gr.inputs.Dropdown(choices=fluid_options, default='CO2', label='工质'),
    ],
  outputs=[
        gr.Plot(label='T-s diagram'), 
        gr.Plot(label="Condition table"),
    ],
  title='Thermodynamics System Visualizer',
  description='Visualize the T-s diagram and condition table of Brayton cycle.'
)

demo2 = gr.Interface(
    fn=find_max,
    inputs=[
        gr.inputs.Dropdown(choices=fluid_options, default='CO2', label='工质'),
        gr.inputs.Number(default=900, label='最高温度 (K)'),
        gr.inputs.Number(default=15, label='最低压强 (MPa)'),
        gr.inputs.Number(default=30, label='最高压强 (MPa)'),
    ],
    outputs=[
        gr.Plot()
    ],
    title='Find the maximum efficiency of Brayton cycle',
    description='Find the maximum efficiency of Brayton cycle with given maximum temperature(It takes about 3 minutes).'
)


demo3 = gr.Interface(
    fn=heat_exchanger,
    inputs=[
        gr.inputs.Dropdown(choices=fluid_options, default='CO2', label='工质'),
        gr.inputs.Slider(minimum=0.5, maximum=1, default=0.7, label='分流比'),
        gr.inputs.Number(default=20, label='最大压强 (MPa)'),
        gr.inputs.Number(default=900, label='最高温度 (K)'),
        gr.inputs.Number(default=277, label='蒸汽发生器功率 (kW)'),
        gr.inputs.Number(default=4000, label='LTR换热系数 (W/m^2/K)'),
        gr.inputs.Number(default=4000, label='HTR换热系数 (W/m^2/K)'),
        gr.inputs.Checkbox(default=False, label='是否画出换热器内部温度变化曲线（大约耗时30秒）')
    ],
    outputs=[
        gr.outputs.Textbox(label='LTR换热面积 (m^2)'),
        gr.outputs.Textbox(label='HTR换热面积 (m^2)'),
        gr.Plot(label='LTR'),
        gr.Plot(label='HTR')
    ],
    title='Heat Exchanger',
    description='Calculate the heat exchanger of Brayton cycle.'
)

demo = gr.TabbedInterface([demo1, demo2, demo3],['system visualizer', 'maximum efficiency with given T_max', 'heat exchanger'])

if __name__ == '__main__':
    app = demo.launch(share=True, show_error=True)
    CORS(app)

