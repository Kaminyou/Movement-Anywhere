import login_demo from '../demo/login_demo.png'
import submit_demo from '../demo/login_demo.png'
import uploading_demo from '../demo/uploading_demo.png'
import computing_demo from '../demo/computing_demo.png'
import result_demo from '../demo/result_demo.png'
import video_demo from '../demo/video_demo.png'

function DemoPage({ token }) {

  return (
    <div className="padding-block">
    <div className="container">
      <div className="row">
      <div className="col-md-10">
      <h2>User Guideline</h2>
          <p>In our system, we set 5 categories of users: admin, manager, general, researcher, and guest. Before login, you are the guest and can only see the demo page. Admins can log in to create manager or general accounts. Managers are able to create general accounts. Please ask them to help you create an account first.</p>
          <h3>Usage</h3>
          <p>Step 1: Please log in with your account.</p>
          <img src={login_demo} class="img-responsive black-frame space-medium" alt="Login Demo" />
          <p>Step 2: Click the UPLOAD, fill the information, and select your video to submit to our server.</p>
          <img src={submit_demo} class="img-responsive black-frame space-medium" alt="Submit Demo" />
          <p>Step 3: The uploading process will be shown. Don't close the page until it has been finished.</p>
          <img src={uploading_demo} class="img-responsive black-frame space-medium" alt="Uploading Demo" />
          <p>Step 4: The inference job will be consumed by our server, please wait for its computation (see the status).</p>
          <img src={computing_demo} class="img-responsive black-frame space-medium" alt="Computing Demo" />
          <p>Step 5: After it has been done, you can click the DASHBOARD page and see your gait parameters.</p>
          <img src={result_demo} class="img-responsive black-frame space-medium" alt="Result Demo" />
          <p>Step 6: You can further check the video of each record.</p>
          <img src={video_demo} class="img-responsive black-frame space-medium" alt="Video Demo" />
      </div>
      </div>
    </div>
  </div>
  )
}

export default DemoPage